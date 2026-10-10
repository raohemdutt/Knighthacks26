"""Raw health data never goes to the Circle (K9): a watch's night is not shared, only the day's habit summary."""

import asyncio
import json
from datetime import timedelta

from bus import EventBus
from contexts.circle.application.protocol import busy
from contexts.circle.application.sharing import HABIT_FIELDS, CircleSharing
from contexts.circle.infrastructure.neon import NeonDb, NeonShares
from contexts.circle.infrastructure.outbox import CircleOutbox
from contexts.circle.tests.neon_helpers import FakeNeon
from contexts.agent.application.memory_port import FakeMemory
from contexts.habits.tests.helpers import at, night, sit, tracker
from contracts.memory import Item

HEART = ("bpm", "HRV", "HR ", "breathing", "core", "deep", "REM", "awake", "in bed", "asleep", "PM", "AM")


def wire(tmp_path, bus: EventBus):
    """Habits and the opted-in sharing on one bus, subscribed the way contexts/gateway/bootstrap.py does it."""
    sharing = CircleSharing(CircleOutbox(tmp_path / "circle.sqlite"), NeonShares(NeonDb(FakeNeon().connect)), "team", "dom", opted_in=True)
    habits = tracker()
    bus.subscribe("connector.item.synced", habits.on_item, replay=True)
    bus.subscribe("perception.zone", habits.on_zone_event, replay=True)
    bus.subscribe("habits.day.closed", habits.on_day_closed, replay=True)
    bus.subscribe("habits.day.closed", sharing.on_day_closed)
    bus.subscribe("impact.count.updated", sharing.on_count_updated)
    bus.subscribe("play.round.scored", sharing.on_round_scored)
    return habits, sharing


def test_sharing_listens_to_summaries_counts_and_scores_and_to_no_connector_item():
    assert {name for name in dir(CircleSharing) if name.startswith("on_")} == {"on_day_closed", "on_count_updated", "on_round_scored"}


def test_a_health_item_queues_no_share_and_the_closed_day_shares_the_summary_only(tmp_path):
    async def run():
        bus = EventBus(tmp_path / "events")
        habits, sharing = wire(tmp_path, bus)
        item = night(at(12, "09:00"), 12, "7:28", resting_hr=47)
        assert "resting HR 47 bpm" in item.summary and "HRV 61 ms" in item.summary  # the item itself has the heart numbers
        await bus.publish_all(sit(11, "21:00", "23:30") + sit(12, "08:30", "08:40"))
        before = sharing.outbox.pending()  # the 11th closed when the 12th began: its seven summaries
        await bus.publish(item)
        assert sharing.outbox.pending() == before and len(before) == 7  # a health item is not a share

        await bus.publish_all(habits.flush(at(13, "00:05")))  # the day closes: this is what leaves
        rows = sharing.outbox.pending()
        assert {r.kind for r in rows} == {"habit_summary"} and all(set(r.payload) == HABIT_FIELDS for r in rows)
        [sleep] = [r.payload for r in rows if r.payload["habit_id"] == "sleep" and r.payload["day"] == "2026-10-12"]
        assert sleep == {"habit_id": "sleep", "day": "2026-10-12", "met": True, "value": 7.47, "streak": 1}
        outbox = json.dumps([r.payload for r in rows])
        assert not [word for word in HEART if word in outbox]  # no stage, no bed time, no heart number

        pushed = sharing.sync(at(13, "00:06"))
        assert len(pushed) == len(rows) and sharing.outbox.pending() == []

    asyncio.run(run())


def test_without_consent_not_even_the_summary_of_a_watch_night_is_queued(tmp_path):
    async def run():
        bus = EventBus(tmp_path / "events")
        habits, sharing = wire(tmp_path, bus)
        sharing.opted_in = False
        await bus.publish(night(at(12, "09:00"), 12, "5:05"))
        await bus.publish_all(habits.flush(at(13, "00:05")))
        assert sharing.outbox.pending() == [] and sharing.sync(at(13, "00:06")) == []

    asyncio.run(run())


def test_the_circle_agent_reads_calendar_events_from_memory_and_never_a_health_item():
    wake = at(12, "07:26")
    memory = FakeMemory(items=[Item(connector="health", kind="health", external_id="sleep:2026-10-12", title="Sleep 7 h 28 min",
                                    when=wake, summary="asleep 7 h 28 min · resting HR 53 bpm")])
    assert busy(memory, wake - timedelta(hours=1), wake + timedelta(hours=1)) is False  # a night is not a calendar event
