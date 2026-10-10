from datetime import UTC, datetime, timedelta

from contexts.impact.domain.waste_watch import WasteWatch
from contexts.play.domain.pathfinding import SortingRules
from contracts.impact import CountUpdated, ItemDisposed, ItemSeen
from contracts.perception import ObservationRecorded, ZoneEvent
from contracts.types import Box, Track

T0 = datetime(2026, 10, 11, 12, tzinfo=UTC)
BOX = Box(x1=0, y1=0, x2=0.1, y2=0.1)


def obs(at, *tracks):
    return ObservationRecorded(ts=T0 + timedelta(minutes=at), camera_id="cam-desk",
                               tracks=[Track(track_id=i, label=label, confidence=0.9, box=BOX) for i, label in tracks])


def binned(at, track_id, label, zone):
    return ZoneEvent(ts=T0 + timedelta(minutes=at), camera_id="cam-desk", track_id=track_id, label=label, zone_id=zone, kind="enter")


def test_items_counted_once_and_disposal_judged_by_the_rules():
    w = WasteWatch(SortingRules.load())
    out = w.on_observation(obs(0, (1, "bottle"), (2, "cup"), (3, "person")))
    assert [type(e) for e in out] == [ItemSeen, ItemSeen, CountUpdated]
    assert w.on_observation(obs(1, (1, "bottle"))) == []  # same track: not counted twice
    [done, counts] = w.on_zone_event(binned(2, 1, "bottle", "recycling"))
    assert isinstance(done, ItemDisposed) and done.correct and done.bin == "recycling"
    [wrong, counts] = w.on_zone_event(binned(3, 2, "cup", "recycling"))
    assert not wrong.correct
    assert counts.model_dump(include={"items_seen", "recyclable_seen", "recycled", "contamination", "diversion_rate"}) == {
        "items_seen": 2, "recyclable_seen": 1, "recycled": 1, "contamination": 1, "diversion_rate": 1.0}
    assert w.on_zone_event(binned(4, 1, "bottle", "trash")) == []  # already disposed


def test_before_and_after_nudges():
    w = WasteWatch(SortingRules.load(), nudges_on_at=T0 + timedelta(hours=1))
    w.on_observation(obs(0, (1, "can"), (2, "bottle")))
    w.on_zone_event(binned(5, 1, "can", "trash"))
    w.on_zone_event(binned(6, 2, "bottle", "trash"))
    w.on_observation(obs(70, (3, "can")))
    w.on_zone_event(binned(75, 3, "can", "recycling"))
    ba = w.before_after()
    assert ba["before"]["diversion_rate"] == 0.0 and ba["after"]["diversion_rate"] == 1.0
