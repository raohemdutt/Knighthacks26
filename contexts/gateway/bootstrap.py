"""Composition root: builds every context and wires it to the bus.

Settings come from the environment (.env), with laptop-friendly defaults:
no model server means FakeRuntime, and until Memory's read API (M4) lands the
agent reads the golden fixture so the whole loop can be demoed end to end.

Replay rule: only handlers that rebuild state subscribe with replay=True.
Anything that sends, writes or calls a model runs on live events only.
"""

import asyncio
from urllib.parse import urlparse
import contextlib
import logging
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

import yaml

from bus import EventBus
from contexts.agent.application.loop import ODD_HOURS_DESCRIPTION, ODD_HOURS_RULE, AgentLoop
from contexts.agent.application.planner import Planner
from contexts.agent.application.qa import QA, SleepSchedule
from contexts.agent.application.web_port import HubWebTools
from contexts.agent.domain.rules import AlertHistory, RuleEvaluator, load_rules
from contexts.capture.application.retention import prune_frames
from contexts.capture.application.service import CaptureService
from contexts.circle.application.delivery import AgreedEventDelivery
from contexts.circle.application.presence import PresenceBoard
from contexts.circle.application.protocol import CircleAgent, InMemoryCircleStore
from contexts.circle.application.sharing import CircleOffline, CircleSharing
from contexts.circle.application.presence_sync import PresenceSync
from contexts.circle.infrastructure.neon import CircleBoard, NeonCircleStore, NeonDb, NeonPresence, NeonShares
from contexts.circle.infrastructure.outbox import CircleOutbox
from contexts.connectors.application.hub import ConnectorHub
from contexts.connectors.domain.spec import load_specs
from contexts.day.application.loop import DayLoop, day_phone
from contexts.connectors.infrastructure.mcp_client import McpConnector
from contexts.learning.application.board import LearningBoard
from contexts.learning.application.export import DatasetExporter
from contexts.memory.application.embedder import embedder_from
from contexts.memory.application.indexer import VectorIndexer
from contexts.memory.application.projection import MemoryProjection
from contexts.memory.application.reader import SqliteMemory
from contexts.memory.application.summarizer import EpisodeSummarizer
from contexts.memory.fixture import as_of, fill_store
from contexts.memory.fixture import load as load_memory_fixture
from contexts.memory.infrastructure.store import MemoryStore
from contexts.habits.application.nudges import Nudger
from contexts.habits.domain.config import load_config as load_habits
from contexts.habits.domain.members import MemberHabits
from contexts.habits.domain.tracker import HabitTracker
from contexts.impact.domain.waste_watch import WasteWatch
from contexts.play.domain.game import SortIt
from contexts.perception.application.identity import IdentityService
from contexts.perception.application.service import PerceptionService
from contexts.perception.infrastructure.identity_store import IdentityStore
from contexts.perception.infrastructure.recognizer import build_recognizer
from contexts.perception.domain.zones import ZoneEngine
from contexts.play.dayrun.service import DayRun
from contexts.play.domain.pathfinding import SortingRules, ZoneMap
from contexts.play.domain.places import PlaceSet
from contexts.vault import KeyRing, Policy, Vault, Wiper
from contexts.workspace.domain.approvals import Approvals
from contexts.workspace.infrastructure.ntfy import Ntfy, PhoneNotifier, send_message_executor
from contracts.types import ModelVersion
from runtime import FakeRuntime
from runtime.registry import build_runtime

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
    # Canned memory loaded into an empty store: the golden night by default, fixtures/memory/memory.yaml for
    # the full demo night, "" on a box that should only hold what its cameras saw.
    memory_fixture: str = os.environ.get("ATLAS_MEMORY_FIXTURE", "fixtures/golden/memory.yaml")
    # Move the canned story so it ends now, for demos on the real clock ("what happened overnight?").
    memory_fixture_now: bool = os.environ.get("ATLAS_MEMORY_FIXTURE_NOW", "0") == "1"


@dataclass
class Atlas:
    settings: Settings
    bus: EventBus
    qa: QA
    approvals: Approvals
    hub: ConnectorHub
    loop: AgentLoop
    zones: list[str] = field(default_factory=list)
    rules: RuleEvaluator | None = None  # GET /api/rules, the Rules screen's switches (X6)
    alerts: AlertHistory | None = None  # GET /api/alerts
    game: SortIt | None = None
    waste: WasteWatch | None = None
    habits: HabitTracker | None = None
    nudger: Nudger | None = None
    zone_engine: ZoneEngine | None = None
    circle: CircleAgent | None = None
    presence: PresenceBoard | None = None
    sharing: CircleSharing | None = None
    memory: SqliteMemory | None = None
    projection: MemoryProjection | None = None
    presence_sync: PresenceSync | None = None
    board: CircleBoard | None = None
    places: PlaceSet | None = None
    dayrun: DayRun | None = None
    day: DayLoop | None = None  # the Day loop (D1): plan, live status, review
    vault: Vault | None = None
    policy: Policy | None = None
    wiper: Wiper | None = None
    learning: LearningBoard | None = None  # where each taught task is (X10)
    capture: CaptureService | None = None  # cameras -> detector -> observations, only where there is a detector (C6)
    identity: IdentityService | None = None  # opt-in owner recognition (P9); off by default
    telemetry: object | None = None  # TelemetryService when ATLAS_TELEMETRY_SOURCE is set; off by default
    tasks: list[asyncio.Task] = field(default_factory=list)
    _last_retention: datetime | None = None
    _offline: set[str] = field(default_factory=set)  # tick jobs whose Neon is down (logged once)

    async def start(self) -> None:
        if self.settings.replay:
            n = await self.bus.replay_into()
            log.info("replayed %d events", n)
        self.tasks.append(asyncio.create_task(self._ticks()))
        if self.telemetry:  # after replay, so the projection is rebuilt before new samples arrive
            self.tasks.append(asyncio.create_task(self.telemetry.run()))
        if self.identity:  # after replay: say the switch and who is enrolled if the log does not have it
            await self.bus.publish_all(self.identity.announce())
        if self.capture:
            self.capture.start(asyncio.get_running_loop())
            if self.identity:  # one recognition hook per camera pipeline; does nothing while switched off
                self.identity.attach(self.capture.pipelines, asyncio.get_running_loop(), self.bus)

    async def _ticks(self) -> None:
        while True:
            now = datetime.now(UTC)
            jobs = [
                ("approvals", lambda: self.approvals.expire(now)),
                # close finished days; nudge on habits about to slip
                ("habits", (lambda: self.habits.flush(now) + self.nudger.check(now)) if self.habits else None),
                ("zones", (lambda: self.zone_engine.flush(now)) if self.zone_engine else None),
                ("episodes", (lambda: self.projection.flush(now)) if self.projection else None),  # close quiet episodes
                ("circle", (lambda: asyncio.to_thread(self.circle.poll)) if self.circle else None),  # network I/O with Neon
                ("shares", (lambda: asyncio.to_thread(self.sharing.sync, now)) if self.sharing else None),
                ("presence", (lambda: asyncio.to_thread(self.presence_sync.sync, now)) if self.presence_sync else None),
                ("agent loop", (lambda: self.loop.tick(now)) if self.loop.due(now) else None),
                ("day", (lambda: self.day.tick(now)) if self.day else None),  # the morning plan, status, evening review
                ("connectors", (lambda: self.hub.sync_due(now)) if self.settings.connectors else None),
                ("retention", self._retention(now)),
                ("camera health", (lambda: self.capture.health(now)) if self.capture else None),
            ]
            for name, job in jobs:
                if job is None:
                    continue
                try:  # each job on its own: Neon being offline must not stop alerts or connector syncs
                    out = job()
                    if asyncio.iscoroutine(out) or isinstance(out, asyncio.Future):
                        out = await out
                    await self.bus.publish_all(out or [])
                    self._offline.discard(name)
                except CircleOffline as exc:  # Neon unreachable: one line when it goes down, not a traceback a tick
                    if name not in self._offline:
                        log.warning("tick job %s: the Circle is offline (%s); retrying every %ss", name, exc, TICK_S)
                    self._offline.add(name)
                except Exception:
                    log.exception("tick job %s failed", name)
            await asyncio.sleep(TICK_S)

    def _retention(self, now: datetime):
        """Once an hour, remove saved pictures older than the policy's keep_days (nothing when it is forever)."""
        if not self.policy or not self.policy.keep_days:
            return None
        if self._last_retention and now - self._last_retention < timedelta(hours=1):
            return None
        self._last_retention = now
        async def prune() -> list:
            await asyncio.to_thread(prune_frames, self.settings.data_dir, self.policy.keep_days, now)
            return []  # no events: the tick publishes whatever a job returns

        return prune

    async def stop(self) -> None:
        if self.capture:
            self.capture.stop()
        for t in self.tasks:
            t.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await t
        await self.hub.close()


def build(settings: Settings | None = None) -> Atlas:
    s = settings or Settings()
    agent_cfg = yaml.safe_load((s.config_dir / "agent.yaml").read_text()) or {}
    bus = EventBus(s.data_dir / "events")

    # Runtime (R1): config/models.yaml picks the models; ATLAS_LLM_URL points at a llama-server.
    models_yaml = s.config_dir / "models.yaml"
    if models_yaml.exists():
        runtime = build_runtime(models_yaml, llm_url=s.llm_url)
    elif s.llm_url:
        from runtime.llama import LlamaServerRuntime

        runtime = LlamaServerRuntime(ModelVersion(version=s.model_version, base="Qwen/Qwen3.5-4B"), base_url=s.llm_url)
    else:
        runtime = FakeRuntime()

    # Memory: the golden fixture until Memory's read API (M4) is up.
    store = MemoryStore(s.data_dir / "memory.sqlite")
    memory = SqliteMemory(store)
    projection = MemoryProjection(store)
    if s.memory_fixture and memory.stats().episodes == 0:
        fixture = load_memory_fixture(s.memory_fixture)
        fill_store(store, as_of(fixture, datetime.now(UTC)) if s.memory_fixture_now else fixture)

    def load_keyframe(ref: str) -> bytes:  # frames:<camera>:<ts_ms> -> data/frames/<camera>/<ts_ms>.jpg
        _, camera, ts_ms = ref.split(":", 2)
        return (s.data_dir / "frames" / camera / f"{ts_ms}.jpg").read_bytes()

    # Connectors (K1) and the browser as agent web tools (K7).
    specs = load_specs(s.config_dir / "connectors.yaml") if s.connectors else []
    hub = ConnectorHub(specs, factory=McpConnector)
    web = HubWebTools(hub) if "browser" in hub.specs else None

    qa = QA(runtime, memory, clock=lambda: datetime.now(UTC), sleep=SleepSchedule.load(s.config_dir / "agent.yaml"),
            web=web, subreddit=agent_cfg.get("subreddit", "ucf"), load_keyframe=load_keyframe)
    planner = Planner(qa, hub)
    if "places" in hub.specs:  # K8: "what should I eat?" uses real nearby places
        planner.places = lambda kind: hub.call_tool("places.nearby", {"kind": kind})
    qa.planner = planner
    rules = RuleEvaluator(load_rules(s.config_dir / "rules.yaml"), builtin={ODD_HOURS_RULE: ODD_HOURS_DESCRIPTION})
    alerts = AlertHistory()

    ntfy = Ntfy(s.ntfy_url, s.ntfy_token) if s.ntfy_url else None
    executors = {
        "connector_write": hub.execute_write,
        "export_training_data": DatasetExporter(
            memory,
            load_keyframe=load_keyframe,
            publish=lambda e: asyncio.get_running_loop().create_task(bus.publish(e)),
            out_dir=s.data_dir / "exports",
        ),
    }
    if ntfy:
        executors["send_message"] = send_message_executor(ntfy, agent_cfg.get("people", {}))
    approvals = Approvals(executors)

    # State that rebuilds from the log (replay=True).
    projection.subscribe(bus)  # episodes, sightings, connector items, hourly notes

    # Privacy Vault (V1-V4): the policy rebuilds from the log. The wipe is live only and not in the
    # background: a replayed wipe would wipe again, and the endpoint reads its report right after publish.
    vault = Vault(KeyRing(s.data_dir))  # no disk access until the first encrypt or decrypt
    policy = Policy(s.data_dir)
    hub.offline = lambda: policy.offline  # offline pauses every connector read and write (V5, H4)
    policy.subscribe(bus)
    # Owner recognition (P9): opt-in, templates encrypted under data/vault/identity/. Unavailable without the SFace model.
    identity = IdentityService(IdentityStore(s.data_dir, vault), build_recognizer())
    bus.subscribe("perception.identity", identity.on_changed, replay=True)
    wiper = Wiper(s.data_dir, memory_db=s.data_dir / "memory.sqlite")
    bus.subscribe("vault.wipe.requested", wiper.on_wipe_requested)
    EpisodeSummarizer(store, runtime).subscribe(bus)  # live only, in the background: one model call per closed episode (M3)
    # Semantic search (M5): off unless models.yaml sets embeddings.text. Live only, in the background, after the projection.
    memory.embedder = embedder_from(runtime, models_yaml)
    VectorIndexer(store, memory.embedder).subscribe(bus)
    bus.subscribe("agent.rule.added", rules.on_rule_added, replay=True)
    bus.subscribe("agent.rule.toggled", rules.on_rule_toggled, replay=True)  # X6: the Rules screen's switches
    bus.subscribe("agent.alert.raised", alerts.on_alert, replay=True)  # GET /api/alerts
    bus.subscribe("perception.zone", rules.on_zone_event, replay=True)
    bus.subscribe("perception.zone", planner.on_zone_event)  # leaving checklist at the door (A8)
    bus.subscribe("agent.action.proposed", approvals.on_proposed, replay=True)
    bus.subscribe("workspace.action.completed", approvals.on_completed, replay=True)
    # Live only: model calls, executors, phone pushes.
    bus.subscribe("experience.question.asked", bus.in_background(qa.on_question))  # model calls take seconds
    bus.subscribe("connector.item.synced", bus.in_background(planner.on_item))  # exam within 48 h -> plan (A8)
    bus.subscribe("experience.approval.decided", bus.in_background(approvals.on_decided))  # executors do I/O
    if ntfy:
        if urlparse(s.public_url).hostname in ("127.0.0.1", "localhost", "::1"):
            # Phones reach the box on the LAN; Approve/Deny buttons on a phone would POST to the phone itself.
            log.warning("ATLAS_PUBLIC_URL is %s: set it to the Jetson's LAN address (http://<jetson-ip>:8000) "
                        "or the approval buttons on phones won't work", s.public_url)
        phones = PhoneNotifier(ntfy, s.ntfy_topic, s.public_url)
        bus.subscribe("agent.alert.raised", phones.on_alert)
        bus.subscribe("habits.nudge.raised", phones.on_nudge)
        bus.subscribe("agent.plan.created", phones.on_plan)
        bus.subscribe("workspace.approval.requested", phones.on_approval)

    zones_file = s.config_dir / "zones.yaml"
    if zones_file.exists():  # {camera: {zone: polygon}}
        zones = sorted({z for cam in (yaml.safe_load(zones_file.read_text()) or {}).values() for z in cam})
    else:
        zones = agent_cfg.get("zones", ["door", "desk"])
    loop = AgentLoop(memory, qa.sleep, summarize=runtime.summarize, notify=agent_cfg.get("notify_on_package"),
                     enabled=rules.enabled)
    bus.subscribe("agent.alert.raised", loop.on_alert)  # live only: proposes a message

    # Perception (P3, P4, P5, P7): tracks -> zone events, captions and desk object lists.
    zone_engine = None
    if zones_file.exists():
        perception = PerceptionService.load(zones_file, runtime, load_keyframe, **(agent_cfg.get("perception") or {}))
        perception.subscribe(bus)  # zones replay; captions and desk object lists are live only, in the background
        zone_engine = perception.engine  # the ticker flushes it

    # Sort It (S1, S2): needs the bin zones and sorting rules in config.
    game = None
    play_cfg = agent_cfg.get("play", {})
    camera = play_cfg.get("camera", "cam-door")
    sorting = s.config_dir / "sorting.yaml"
    if zones_file.exists() and sorting.exists() and camera in (yaml.safe_load(zones_file.read_text()) or {}):
        game = SortIt(ZoneMap.load(camera, zones_file), SortingRules.load(sorting), start_xy=tuple(play_cfg.get("start_xy", (0.75, 0.75))))
        bus.subscribe("perception.faces", game.on_faces)
        bus.subscribe("perception.zone", game.on_zone_event)
        bus.subscribe("play.round.scored", game.on_scored, replay=True)  # state only: rounds played before a restart

    # Waste Watch (E1, E2): counts rebuild from the log, so they survive restarts.
    waste = None
    if sorting.exists():
        nudges_on = (agent_cfg.get("impact") or {}).get("nudges_on_at")
        waste = WasteWatch(SortingRules.load(sorting), nudges_on_at=datetime.fromisoformat(nudges_on) if nudges_on else None)
        bus.subscribe("perception.observation", waste.on_observation, replay=True)
        bus.subscribe("perception.zone", waste.on_zone_event, replay=True)
        bus.subscribe("habits.nudge.raised", waste.on_nudge, replay=True)  # state only: when nudges started (E2)

    # Habits (H1, H2): the daily picture rebuilds from the log; nudges are live only.
    habits = nudger = None
    habits_file = s.config_dir / "habits.yaml"
    if habits_file.exists():
        habits_cfg = load_habits(habits_file, s.config_dir / "agent.yaml")
        habits = MemberHabits(HabitTracker(habits_cfg), nudger_for=lambda t: Nudger(habits_cfg, t))  # P9: per enrolled person
        nudger = Nudger(habits_cfg, habits.base)
        bus.subscribe("perception.identity", habits.on_identity, replay=True)  # recognition on/off, who is enrolled
        bus.subscribe("perception.person", habits.on_person, replay=True)  # track N on camera C is member M
        bus.subscribe("perception.zone", habits.on_zone_event, replay=True)
        bus.subscribe("perception.scene", habits.on_scene, replay=True)
        bus.subscribe("perception.observation", habits.on_observation, replay=True)  # when the desk camera was reporting
        bus.subscribe("vault.policy.changed", habits.on_policy, replay=True)  # ... and when it was paused
        bus.subscribe("connector.item.synced", habits.on_item, replay=True)
        bus.subscribe("impact.item.disposed", habits.on_disposed, replay=True)
        bus.subscribe("habits.day.closed", habits.on_day_closed, replay=True)
        bus.subscribe("habits.nudge.raised", habits.on_nudge, replay=True)  # state only: latest nudge, cooldown
        bus.subscribe("habits.signal.recorded", nudger.on_signal)
        bus.subscribe("habits.day.closed", nudger.on_day_closed)
        planner.habits = habits  # A8: plans and food answers use today's habit numbers

    # Circle (G1): friends' Atlases negotiate through agent_messages. The store is in-memory until
    # the Neon store from N1 is plugged in here (same CircleStore port).
    circle = sharing = neon = board = None
    circle_cfg = agent_cfg.get("circle") or {}
    if circle_cfg.get("member_id") and circle_cfg.get("circle_id"):
        # Neon (N1) when NEON_DATABASE_URL is set; otherwise two Atlases can still negotiate in one process.
        names = circle_cfg.get("names") or {}  # member id -> display name; also the circle's member list
        neon = NeonDb(names=names) if os.environ.get("NEON_DATABASE_URL") else None
        store = NeonCircleStore(neon) if neon else InMemoryCircleStore()
        circle = CircleAgent(circle_cfg["member_id"], circle_cfg["circle_id"], store, memory,
                             clock=lambda: datetime.now(UTC), members=list(names))
        bus.subscribe("circle.message.received", circle.on_message, replay=True)  # state only: never answer twice
        bus.subscribe("circle.event.agreed", circle.on_agreed, replay=True)  # state only: never agree twice
        approvals.executors["circle_share"] = circle.share_executor
        # G1 step 4: once everyone says yes, propose the calendar invite and the group chat message.
        delivery = AgreedEventDelivery(hub, circle_cfg["member_id"], qa.sleep.tz, emails=circle_cfg.get("emails") or {},
                                       chat_group=circle_cfg.get("chat_group", "friends"), public_url=s.public_url)
        bus.subscribe("circle.event.agreed", delivery.on_agreed)  # live only: proposes writes
        if neon:
            # Summaries leave the box only when the member opted in; they queue locally first.
            sharing = CircleSharing(CircleOutbox(s.data_dir / "circle.sqlite"), NeonShares(neon), circle_cfg["circle_id"],
                                    circle_cfg["member_id"], opted_in=bool(circle_cfg.get("share_summaries")))
            bus.subscribe("habits.day.closed", sharing.on_day_closed)  # live only: they write
            bus.subscribe("impact.count.updated", sharing.on_count_updated)
            bus.subscribe("play.round.scored", sharing.on_round_scored)
            board = CircleBoard(neon)  # leaderboards for the Circle screen and E2

    # Friend presence (G5): opt-in, named places only, expiring.
    presence = None
    if circle:
        presence = PresenceBoard(circle_cfg["circle_id"], circle_cfg["member_id"],
                                 places=agent_cfg.get("places", ["Our table", "Library", "Dorm", "Student Union", "Gym"]),
                                 opted_in=bool(circle_cfg.get("share_presence")))
        bus.subscribe("circle.place.shared", presence.on_shared, replay=True)
        bus.subscribe("perception.zone", presence.on_zone_event)
        planner.presence = presence

    # Presence delivery (G5): my places to Neon, friends' places back as events.
    presence_sync = None
    if presence and neon:
        presence_sync = PresenceSync(NeonPresence(neon), circle_cfg["circle_id"], circle_cfg["member_id"])
        bus.subscribe("circle.place.shared", presence_sync.on_shared)  # live only: it writes

    # Place packs (S5): shipped packs in config/places, uploads in data/places (later wins).
    try:
        places = PlaceSet.load(s.config_dir / "places", s.data_dir / "places")
    except ValueError as exc:
        log.warning("place packs not loaded: %s", exc)
        places = None
    if places:  # A8 leaving checklist: real walking time from home to the next event
        planner.place_set = places
        planner.home_place = (agent_cfg.get("leaving") or {}).get("home_place")

    # Day Run (S4, S6): play tomorrow on the place packs before living it. No model calls.
    dayrun = DayRun(places, memory=memory, habits=habits, presence=presence, hub=hub,
                    config_dir=s.config_dir, data_dir=s.data_dir) if places else None

    # The Day loop (D1): the morning plan, the live status and the evening review, unasked. Plans, statuses and
    # reviews rebuild from the log; deciding and proposing cards is live only.
    day = None
    if dayrun:
        day = DayLoop(dayrun, places, config_dir=s.config_dir, habits=habits, memory=memory, waste=waste, presence=presence, hub=hub)
        day.subscribe(bus)
        if ntfy:
            bus.subscribe("day.", day_phone(ntfy, s.ntfy_topic))  # the plan, a re-plan and the review on the phone

    # Learning board (X10): the Teach screen's view of tasks, runs and deployed models; rebuilt from the log.
    learning = LearningBoard()
    learning.subscribe(bus)

    # Cameras (C1, P1, P2, P6): config/cameras.yaml; None unless a detector engine is on this machine.
    capture = CaptureService.load(s.config_dir / "cameras.yaml", bus, models_yaml=models_yaml, data_dir=s.data_dir, policy=policy)

    # Telemetry: per-node power samples, request windows and training steps as events, folded into a projection
    # the log rebuilds. Off unless ATLAS_TELEMETRY_SOURCE names a sampler ("fake" replays recorded samples).
    telemetry = None
    if os.environ.get("ATLAS_TELEMETRY_SOURCE"):
        from contexts.telemetry.wiring import build_telemetry

        telemetry = build_telemetry(os.environ["ATLAS_TELEMETRY_SOURCE"], bus)

    return Atlas(settings=s, bus=bus, qa=qa, approvals=approvals, hub=hub, loop=loop, zones=zones, rules=rules, alerts=alerts, game=game,
                 waste=waste, circle=circle, presence=presence, zone_engine=zone_engine, sharing=sharing,
                 memory=memory, projection=projection, habits=habits, nudger=nudger,
                 presence_sync=presence_sync, board=board, vault=vault, policy=policy, wiper=wiper,
                 places=places, dayrun=dayrun, day=day, capture=capture, identity=identity, learning=learning,
                 telemetry=telemetry)
