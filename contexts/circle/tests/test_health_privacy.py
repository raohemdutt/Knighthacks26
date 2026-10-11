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


# --- the check-in (contexts/day/application/checkin.py) is box-only: `day.checkin.*` never reaches the Circle ---


def checkin_events():
    """A check-in and its answer, from the contract samples (invented values: nothing measured on anybody)."""
    from contracts import parse_event
    from tests.test_contract_samples import SAMPLES

    return [parse_event({"type": t, "ts": at(12, "15:40").isoformat(), **SAMPLES[t]}) for t in ("day.checkin.raised", "day.checkin.answered")]


def test_a_check_in_queues_nothing_for_the_circle_and_nothing_of_it_is_synced(tmp_path):
    from contracts.day import BOX_ONLY_EVENTS

    raised, answered = checkin_events()
    assert {raised.type, answered.type} <= BOX_ONLY_EVENTS  # the set also holds `day.optional.answered` (tested below)
    said = [s.summary for s in raised.signals] + [s.detail for s in raised.signals] + [s.text for s in raised.suggestions] + [raised.text]
    assert any("Heart rate" in line for line in said) and any("slept" in line for line in said)  # it really is about a body

    async def run():
        bus = EventBus(tmp_path / "events")
        habits, sharing = wire(tmp_path, bus)
        await bus.publish_all(sit(11, "21:00", "23:30") + sit(12, "08:30", "08:40"))
        before = [r.payload for r in sharing.outbox.pending()]
        await bus.publish_all([raised, answered])
        assert [r.payload for r in sharing.outbox.pending()] == before  # a check-in is not a share
        await bus.publish_all(habits.flush(at(13, "00:05")))
        rows = sharing.outbox.pending()
        assert {r.kind for r in rows} == {"habit_summary"} and all(set(r.payload) == HABIT_FIELDS for r in rows)
        leaving = json.dumps([r.payload for r in rows])
        assert not [line for line in said if line in leaving]
        assert not [word for word in ("strained", "Heart rate", "heart rate", "No drink", "Drink some", "checkin", "check-in") if word in leaving]
        pushed = sharing.sync(at(13, "00:06"))
        assert len(pushed) == len(rows)

    asyncio.run(run())


# --- an accepted optional event (contexts/day/application/optional.py): the friends its tags name stay on the box ---


def test_an_accepted_optional_event_names_no_friend_to_the_circle(tmp_path):
    """Saying yes to "Friends' dinner ... with Riley, Sam, Alex" and approving the card makes it a block of the day.
    The block has the title, the time and the place; who is going stays in the calendar item. What the Circle is sent
    (the closed day's habit summaries) names neither a friend nor the event."""
    from contexts.day.tests.test_optional_events import DINNER, FRIENDS, afternoon, said_yes
    from contexts.day.tests.test_optional_events import at as on
    from contexts.play.domain.places import PlaceSet
    from contracts.day import BOX_ONLY_EVENTS

    b = afternoon(PlaceSet.load("config/places"), tmp_path)  # the Day loop's box, Monday the 12th at 4 PM
    habits, sharing = wire(tmp_path, b.bus)  # ... with Habits and the opted-in sharing on the same bus
    body = said_yes(b, DINNER, on(16, 1))
    b.approve(body["action_id"], on(16, 2))
    assert b.optional.accepted == {DINNER} and [m.title for m in b.plan().moves if m.origin == "optional"] == ["Friends' dinner"]
    b.publish(*sit(12, "16:05", "16:35"))
    b.publish(*habits.flush(at(13, "00:05")))  # the day closes: this is what leaves
    rows = sharing.outbox.pending()
    assert rows and {r.kind for r in rows} == {"habit_summary"} and all(set(r.payload) == HABIT_FIELDS for r in rows)
    leaving = json.dumps([r.payload for r in rows])
    assert not [word for word in (*FRIENDS, "dinner", "nearby", "optional", "63 South", "GroupMe") if word in leaving]
    # nothing a plan carries could take a name out either: no block's id, title or place holds one
    plans = [e for e in b.out if e.type in ("day.plan.proposed", "day.replan.proposed")]
    assert len(plans) >= 2 and not [m for e in plans for m in e.moves if [n for n in FRIENDS if n in f"{m.id} {m.title} {m.place_name}"]]
    answers = [e for e in b.out if e.type == "day.optional.answered"]
    assert [e.id for e in answers] == [DINNER] and {e.type for e in answers} <= BOX_ONLY_EVENTS
    assert not [n for n in FRIENDS if n in answers[0].model_dump_json()]  # which event, never who


def test_nothing_that_writes_to_the_circle_knows_the_day_context_or_a_check_in():
    """Structural: the code that leaves the box (the Circle, the League's sync) imports nothing of the Day loop and
    names no `day.` event, so no handler there can ever be handed a check-in."""
    import re
    from pathlib import Path

    from contracts.day import BOX_ONLY_EVENTS

    root = Path(__file__).resolve().parents[3]
    for folder in ("contexts/circle", "contexts/play/league"):
        for path in (root / folder).rglob("*.py"):
            if "tests" in path.parts:
                continue
            text = path.read_text(encoding="utf-8")
            assert not re.search(r"contracts\.day|contexts\.day|[\"']day\.|checkin|Checkin", text), path
            assert not [t for t in BOX_ONLY_EVENTS if t in text], path
