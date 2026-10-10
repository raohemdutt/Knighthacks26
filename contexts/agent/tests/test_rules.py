from datetime import UTC, datetime, timedelta

from contexts.agent.domain.rules import RuleEvaluator, load_rules
from contracts.perception import ZoneEvent

T0 = datetime(2026, 10, 10, 12, 0, tzinfo=UTC)


def zone(label="package", kind="dwell", dwell_s=6.0, zone_id="door", at=0):
    return ZoneEvent(
        ts=T0 + timedelta(seconds=at),
        camera_id="cam-door",
        track_id=3,
        label=label,
        zone_id=zone_id,
        kind=kind,
        dwell_s=dwell_s,
        frame_ref=f"frames:cam-door:{at}",
    )


def evaluator():
    return RuleEvaluator(load_rules("config/rules.yaml"))


def test_package_dwell_at_door_fires_with_evidence():
    [alert] = evaluator().on_zone_event(zone())
    assert alert.rule_id == "package-at-door"
    assert alert.message == "Package left at the door"
    assert alert.evidence[0].frame_ref == "frames:cam-door:0"
    assert alert.ts == T0


def test_short_dwell_wrong_label_wrong_zone_do_not_fire():
    ev = evaluator()
    assert ev.on_zone_event(zone(dwell_s=2)) == []
    assert ev.on_zone_event(zone(label="person")) == []
    assert ev.on_zone_event(zone(zone_id="desk")) == []
    assert ev.on_zone_event(zone(kind="enter")) == []


def test_cooldown_uses_event_time():
    ev = evaluator()
    assert len(ev.on_zone_event(zone(at=0))) == 1
    assert ev.on_zone_event(zone(at=60)) == []
    assert len(ev.on_zone_event(zone(at=121))) == 1
