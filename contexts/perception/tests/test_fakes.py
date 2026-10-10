import asyncio
from datetime import UTC, datetime, timedelta

from bus import EventBus
from contexts.agent.domain.rules import RuleEvaluator, load_rules
from contexts.impact.domain.waste_watch import WasteWatch
from contexts.perception.domain.zones import ZoneEngine
from contexts.perception.fakes import can_recycled, observations, package_left, play, script
from contexts.play.domain.pathfinding import SortingRules
from contracts.agent import AlertRaised
from contracts.impact import ItemDisposed
from contracts.perception import ObservationRecorded, ZoneEvent

T0 = datetime(2026, 10, 10, 12, 0, tzinfo=UTC)


def zone_events(obs):
    engine = ZoneEngine.load("config/zones.yaml")
    return [e for o in obs for e in engine.on_observation(o)]


def story(events):
    return [(int((e.ts - T0).total_seconds()), e.camera_id, e.label, e.zone_id, e.kind) for e in events if e.kind != "dwell"]


def test_script_is_deterministic_and_once_a_second_per_camera():
    obs = script(T0)
    assert obs == script(T0)
    assert len(obs) == 62 and obs[0].ts == T0 and obs[-1].ts == T0 + timedelta(seconds=30)
    assert all(isinstance(o, ObservationRecorded) for o in obs)
    later = script(T0 + timedelta(hours=1))
    assert [o.ts - timedelta(hours=1) for o in later] == [o.ts for o in obs]
    assert [o.tracks for o in later] == [o.tracks for o in obs]


def test_script_tells_the_whole_scene_as_zone_events():
    events = zone_events(script(T0))
    assert story(events) == [
        (0, "cam-door", "person", "door", "enter"),
        (3, "cam-door", "person", "door", "exit"),
        (5, "cam-desk", "person", "desk", "enter"),
        (8, "cam-desk", "keys", "desk", "enter"),
        (10, "cam-door", "package", "door", "enter"),
        (24, "cam-door", "can", "recycling", "enter"),
        (28, "cam-door", "can", "recycling", "exit"),  # dropped in the bin: gone from view, timed out
    ]
    package = [e.dwell_s for e in events if e.label == "package" and e.kind == "dwell"]
    assert package == [3.0, 6.0, 9.0, 12.0, 15.0, 18.0]  # still there when the script ends


def test_package_at_door_raises_the_alert_through_the_real_rules():
    rules = RuleEvaluator(load_rules("config/rules.yaml"))
    alerts = [a for e in zone_events(observations(T0, package_left(at=0, until=20))) for a in rules.on_zone_event(e)]
    [alert] = alerts  # later dwells fall inside the rule's cooldown
    assert isinstance(alert, AlertRaised) and alert.rule_id == "package-at-door"
    assert alert.ts == T0 + timedelta(seconds=6)  # first dwell of at least 5 s


def test_can_is_counted_as_recycled_by_waste_watch():
    waste = WasteWatch(SortingRules.load("config/sorting.yaml"))
    engine = ZoneEngine.load("config/zones.yaml")
    out = []
    for o in observations(T0, can_recycled(at=0), length_s=10):
        out += waste.on_observation(o)
        for e in engine.on_observation(o):
            out += waste.on_zone_event(e)
    [done] = [e for e in out if isinstance(e, ItemDisposed)]
    assert done.label == "can" and done.bin == "recycling" and done.correct
    assert waste.metrics() == {"items_seen": 1, "recyclable_seen": 1, "recycled": 1, "contamination": 0, "diversion_rate": 1.0}


def wire(bus):
    """The same subscriptions as the gateway, for the parts that consume zone events."""
    engine = ZoneEngine.load("config/zones.yaml")
    rules = RuleEvaluator(load_rules("config/rules.yaml"))
    waste = WasteWatch(SortingRules.load("config/sorting.yaml"))
    seen = []
    bus.subscribe("perception.observation", engine.on_observation, replay=True)
    bus.subscribe("perception.observation", waste.on_observation, replay=True)
    bus.subscribe("perception.zone", waste.on_zone_event, replay=True)
    bus.subscribe("perception.zone", rules.on_zone_event, replay=True)
    bus.subscribe("", seen.append, replay=True)
    return engine, waste, seen


def test_play_on_the_bus_reaches_rules_and_waste_watch(tmp_path):
    bus = EventBus(tmp_path)
    engine, waste, seen = wire(bus)
    assert asyncio.run(play(bus, T0)) == 62
    assert [e.rule_id for e in seen if isinstance(e, AlertRaised)] == ["package-at-door"]
    assert waste.metrics()["recycled"] == 1
    assert [e for e in seen if isinstance(e, ZoneEvent)] == zone_events(script(T0))


def test_restart_rebuilds_state_from_the_log_without_repeating_zone_events(tmp_path):
    async def restart(flush_at):
        bus = EventBus(tmp_path)
        engine, waste, seen = wire(bus)
        await bus.replay_into()
        await bus.publish_all(engine.flush(flush_at))  # first tick after the restart
        return engine, waste, [e for e in seen if isinstance(e, ZoneEvent)]

    asyncio.run(play(EventBus(tmp_path), T0))  # observations only: nothing subscribed
    first_bus = EventBus(tmp_path)
    engine, _, _ = wire(first_bus)
    asyncio.run(first_bus.replay_into())
    assert set(engine.stays) == {("cam-desk", 1, "desk"), ("cam-desk", 3, "desk"), ("cam-door", 2, "door")}

    _, _, zone_log = asyncio.run(restart(T0 + timedelta(minutes=5)))
    assert [(e.label, e.kind) for e in zone_log] == [("person", "exit"), ("keys", "exit"), ("package", "exit")]
    assert {e.ts for e in zone_log} == {T0 + timedelta(seconds=33)}
    engine3, _, again = asyncio.run(restart(T0 + timedelta(minutes=10)))
    assert again == zone_log and engine3.stays == {}  # replayed from the log; the flush's copies were deduped
