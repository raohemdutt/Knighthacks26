import asyncio
from datetime import UTC, datetime, timedelta

from bus import EventBus
from contexts.agent.domain.rules import RuleEvaluator, load_rules
from contracts.agent import AlertRaised
from contracts.capture import FramesCaptured
from contracts.perception import ZoneEvent

T0 = datetime(2026, 10, 10, 23, 59, 50, tzinfo=UTC)


def package(at=0):
    return ZoneEvent(
        ts=T0 + timedelta(seconds=at), camera_id="cam-door", track_id=1, label="package", zone_id="door", kind="dwell", dwell_s=6
    )


def test_publish_logs_then_dispatches_by_prefix_and_follow_ups(tmp_path):
    bus = EventBus(tmp_path)
    alerts, everything = [], []
    rules = RuleEvaluator(load_rules("config/rules.yaml"))
    bus.subscribe("perception.zone", rules.on_zone_event)  # sync handler returning events
    bus.subscribe("agent.alert", alerts.append)

    async def record(e):
        everything.append(e.type)

    bus.subscribe("", record)  # async handler, sees everything
    asyncio.run(bus.publish(package()))
    assert [a.rule_id for a in alerts] == ["package-at-door"]
    assert everything == ["perception.zone.event", "agent.alert.raised"]
    lines = (tmp_path / "2026-10-10.jsonl").read_text().splitlines()
    assert len(lines) == 2


def test_dedupe_failures_and_unlogged_types(tmp_path):
    bus = EventBus(tmp_path)
    seen = []

    def boom(e):
        raise RuntimeError("bad handler")

    bus.subscribe("", boom)
    bus.subscribe("", seen.append)
    e = package()
    asyncio.run(bus.publish_all([e, e]))
    assert seen == [e]
    asyncio.run(bus.publish(FramesCaptured(camera_id="cam-door", seq=1, width=1, height=1, motion_score=0)))
    assert len(seen) == 2
    assert sum(len(p.read_text().splitlines()) for p in tmp_path.glob("*.jsonl")) == 1


def test_log_is_split_by_utc_day_and_replays_without_duplicating(tmp_path):
    live = EventBus(tmp_path)
    live.subscribe("perception.zone", RuleEvaluator(load_rules("config/rules.yaml")).on_zone_event)
    asyncio.run(live.publish_all([package(0), package(200)]))  # second one is on the next UTC day
    assert sorted(p.name for p in tmp_path.glob("*.jsonl")) == ["2026-10-10.jsonl", "2026-10-11.jsonl"]

    rebuilt = EventBus(tmp_path)
    alerts = []
    rebuilt.subscribe("perception.zone", RuleEvaluator(load_rules("config/rules.yaml")).on_zone_event, replay=True)
    rebuilt.subscribe("agent.alert", alerts.append, replay=True)
    assert asyncio.run(rebuilt.replay_into()) == 4
    assert len(alerts) == 2 and all(isinstance(a, AlertRaised) for a in alerts)  # the logged ones, not new copies
    assert sum(len(p.read_text().splitlines()) for p in tmp_path.glob("*.jsonl")) == 4


def test_replay_never_reruns_side_effects(tmp_path):
    import asyncio as aio

    from contexts.workspace.domain.approvals import Approvals
    from contracts.agent import ActionProposed
    from contracts.experience import ApprovalDecided

    def wire(bus, sent):
        approvals = Approvals({"send_message": lambda r: sent.append(r.target) or "sent"})
        bus.subscribe("agent.action.proposed", approvals.on_proposed)
        bus.subscribe("experience.approval.decided", approvals.on_decided)

    sent = []
    live = EventBus(tmp_path)
    wire(live, sent)
    aio.run(live.publish_all([
        ActionProposed(ts=T0, action_id="a1", kind="send_message", target="dom", reason="test"),
        ApprovalDecided(ts=T0, action_id="a1", decision="approve"),
    ]))
    assert sent == ["dom"]

    again = []
    rebuilt = EventBus(tmp_path)
    wire(rebuilt, again)
    assert aio.run(rebuilt.replay_into()) == 4  # proposed, requested, decided, completed
    assert again == []


def test_background_handlers_do_not_block_the_bus(tmp_path):
    async def run():
        bus = EventBus(tmp_path)
        order = []

        async def slow(e):
            await asyncio.sleep(0.05)
            order.append("slow done")
            return AlertRaised(alert_id="a", rule_id="r", message="m")

        bus.subscribe("perception.zone", bus.in_background(slow))
        bus.subscribe("agent.alert", lambda e: order.append("alert"))
        await bus.publish(package())
        order.append("publish returned")
        await asyncio.sleep(0.1)
        return order

    assert asyncio.run(run()) == ["publish returned", "slow done", "alert"]
