"""Composition root: builds every context and wires it to the bus.

Settings come from the environment (.env), with laptop-friendly defaults:
no model server means FakeRuntime, and until Memory's read API (M4) lands the
agent reads the golden fixture so the whole loop can be demoed end to end.

Replay rule: only handlers that rebuild state subscribe with replay=True.
Anything that sends, writes or calls a model runs on live events only.
"""

import asyncio
import contextlib
import logging
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import yaml

from bus import EventBus
from contexts.agent.application.planner import Planner
from contexts.agent.application.qa import QA, SleepSchedule
from contexts.agent.application.web_port import HubWebTools
from contexts.agent.domain.rules import RuleEvaluator, load_rules
from contexts.agent.eval import load_fixture
from contexts.connectors.application.hub import ConnectorHub
from contexts.connectors.domain.spec import load_specs
from contexts.connectors.infrastructure.mcp_client import McpConnector
from contexts.learning.application.export import DatasetExporter
from contexts.workspace.domain.approvals import Approvals
from contexts.workspace.infrastructure.ntfy import Ntfy, PhoneNotifier, send_message_executor
from contracts.types import ModelVersion
from runtime import FakeRuntime

log = logging.getLogger("atlas")
TICK_S = 30


@dataclass
class Settings:
    data_dir: Path = Path(os.environ.get("ATLAS_DATA_DIR", "data"))
    config_dir: Path = Path(os.environ.get("ATLAS_CONFIG_DIR", "config"))
    llm_url: str | None = os.environ.get("ATLAS_LLM_URL")  # llama-server, e.g. http://127.0.0.1:8080
    model_version: str = os.environ.get("ATLAS_MODEL_VERSION", "atlas-base")
    ntfy_url: str | None = os.environ.get("NTFY_URL")  # e.g. http://127.0.0.1:8090
    ntfy_token: str | None = os.environ.get("NTFY_TOKEN")
    ntfy_topic: str = os.environ.get("NTFY_TOPIC", "atlas-alerts")
    public_url: str = os.environ.get("ATLAS_PUBLIC_URL", "http://127.0.0.1:8000")
    token: str | None = os.environ.get("ATLAS_TOKEN")  # required on /api writes when set
    connectors: bool = os.environ.get("ATLAS_CONNECTORS", "1") == "1"
    replay: bool = True


@dataclass
class Atlas:
    settings: Settings
    bus: EventBus
    qa: QA
    approvals: Approvals
    hub: ConnectorHub
    tasks: list[asyncio.Task] = field(default_factory=list)

    async def start(self) -> None:
        if self.settings.replay:
            n = await self.bus.replay_into()
            log.info("replayed %d events", n)
        self.tasks.append(asyncio.create_task(self._ticks()))

    async def _ticks(self) -> None:
        while True:
            now = datetime.now(UTC)
            try:
                await self.bus.publish_all(self.approvals.expire(now))
                if self.settings.connectors:
                    await self.bus.publish_all(await self.hub.sync_due(now))
            except Exception:
                log.exception("tick failed")
            await asyncio.sleep(TICK_S)

    async def stop(self) -> None:
        for t in self.tasks:
            t.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await t
        await self.hub.close()


def build(settings: Settings | None = None) -> Atlas:
    s = settings or Settings()
    agent_cfg = yaml.safe_load((s.config_dir / "agent.yaml").read_text()) or {}
    bus = EventBus(s.data_dir / "events")

    # Runtime (R1): the real model if a llama-server is configured.
    if s.llm_url:
        from runtime.llama import LlamaServerRuntime

        runtime = LlamaServerRuntime(ModelVersion(version=s.model_version, base="Qwen/Qwen3.5-4B"), base_url=s.llm_url)
    else:
        runtime = FakeRuntime()

    # Memory: the golden fixture until Memory's read API (M4) is up.
    memory, _, _ = load_fixture()

    # Connectors (K1) and the browser as agent web tools (K7).
    specs = load_specs(s.config_dir / "connectors.yaml") if s.connectors else []
    hub = ConnectorHub(specs, factory=McpConnector)
    web = HubWebTools(hub) if "browser" in hub.specs else None

    qa = QA(runtime, memory, clock=lambda: datetime.now(UTC), sleep=SleepSchedule.load(s.config_dir / "agent.yaml"),
            web=web, subreddit=agent_cfg.get("subreddit", "ucf"))
    planner = Planner(qa, hub)
    qa.planner = planner
    rules = RuleEvaluator(load_rules(s.config_dir / "rules.yaml"))

    ntfy = Ntfy(s.ntfy_url, s.ntfy_token) if s.ntfy_url else None
    executors = {
        "connector_write": hub.execute_write,
        "export_training_data": DatasetExporter(
            memory,
            load_keyframe=lambda ref: (s.data_dir / "frames" / (ref.split(":", 1)[1].replace(":", "/") + ".jpg")).read_bytes(),
            publish=lambda e: asyncio.get_running_loop().create_task(bus.publish(e)),
            out_dir=s.data_dir / "exports",
        ),
    }
    if ntfy:
        executors["send_message"] = send_message_executor(ntfy, agent_cfg.get("people", {}))
    approvals = Approvals(executors)

    # State that rebuilds from the log (replay=True).
    bus.subscribe("perception.zone", rules.on_zone_event, replay=True)
    bus.subscribe("perception.zone", planner.on_zone_event)  # leaving checklist at the door (A8)
    bus.subscribe("agent.action.proposed", approvals.on_proposed, replay=True)
    bus.subscribe("workspace.action.completed", approvals.on_completed, replay=True)
    # Live only: model calls, executors, phone pushes.
    bus.subscribe("experience.question.asked", bus.in_background(qa.on_question))  # model calls take seconds
    bus.subscribe("connector.item.synced", bus.in_background(planner.on_item))  # exam within 48 h -> plan (A8)
    bus.subscribe("experience.approval.decided", bus.in_background(approvals.on_decided))  # executors do I/O
    if ntfy:
        phones = PhoneNotifier(ntfy, s.ntfy_topic, s.public_url)
        bus.subscribe("agent.alert.raised", phones.on_alert)
        bus.subscribe("habits.nudge.raised", phones.on_nudge)
        bus.subscribe("agent.plan.created", phones.on_plan)
        bus.subscribe("workspace.approval.requested", phones.on_approval)

    return Atlas(settings=s, bus=bus, qa=qa, approvals=approvals, hub=hub)
