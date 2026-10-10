"""A plan the Circle agreed is a fixed block of the day, for every attendee, from the log alone."""

import asyncio
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from bus.bus import EventBus
from contexts.agent.application.memory_port import FakeMemory
from contexts.play.dayrun.agreed import AgreedPlans
from contexts.play.dayrun.service import DayRun
from contexts.play.domain.places import PlaceSet
from contracts.circle import EventAgreed, Proposal
from contracts.memory import Item

TZ = ZoneInfo("America/New_York")
NOW = datetime(2026, 10, 10, 9, 30, tzinfo=TZ)


def at(hour: int, minute: int = 0, day: int = 10) -> datetime:
    return datetime(2026, 10, day, hour, minute, tzinfo=TZ)


def agreed(thread: str = "t1", title: str = "COP4533 study session", start: datetime = at(15), hours: float = 2,
           location: str | None = "the Library", attendees: tuple = ("hemdutt", "maya")) -> EventAgreed:
    return EventAgreed(ts=NOW - timedelta(hours=1), circle_id="team", thread_id=thread, attendees=sorted(attendees),
                       created_by="hemdutt", event=Proposal(title=title, start=start, end=start + timedelta(hours=hours),
                                                            location=location, invitees=["maya"]))


@pytest.fixture
def places() -> PlaceSet:
    return PlaceSet.load("config/places")


def dayrun(places, tmp_path, plans: AgreedPlans | None, memory=None) -> DayRun:
    return DayRun(places, memory=memory, data_dir=tmp_path, clock=lambda: NOW, agreed=plans)


def test_an_agreed_plan_is_a_fixed_block_with_its_place_for_both_members(places, tmp_path):
    e = agreed()
    for member in ("hemdutt", "maya"):  # each attendee's box publishes its own copy of the agreement
        plans = AgreedPlans(member)
        plans.on_agreed(e)
        board = dayrun(places, tmp_path / member, plans).board()
        [block] = [b for b in board.blocks if b.id == "circle:t1"]
        assert (block.title, block.start, block.end) == ("COP4533 study session", at(15), at(17))
        assert (block.place, block.place_name, block.activity) == ("ucf.library", "John C. Hitt Library", "study")
        assert all(not (s.start < block.end and block.start < s.end) for s in board.slots)  # no open slot over it


def test_without_the_agreement_the_board_has_no_such_block(places, tmp_path):
    assert [b.id for b in dayrun(places, tmp_path, None).board().blocks if b.id.startswith("circle:")] == []
    assert [b.id for b in dayrun(places, tmp_path, AgreedPlans("hemdutt")).board().blocks if b.id.startswith("circle:")] == []


def test_only_plans_this_member_is_in_and_only_inside_the_window(places, tmp_path):
    plans = AgreedPlans("hemdutt")
    for e in (agreed("mine", "Dinner", at(18, 30), 1, "the Student Union"),
              agreed("theirs", "Movie", at(20), attendees=("maya", "dominick")),
              agreed("past", "Breakfast", at(8, day=9), 1),
              agreed("far", "Trip", at(10, day=13))):
        plans.on_agreed(e)
    blocks = {b.id: b for b in dayrun(places, tmp_path, plans).board().blocks}
    assert [i for i in blocks if i.startswith("circle:")] == ["circle:mine"]
    assert (blocks["circle:mine"].activity, blocks["circle:mine"].place) == ("eat", "ucf.union")


def test_a_plan_the_calendar_already_holds_is_not_there_twice(places, tmp_path):
    """The proposer's calendar connector may sync the invite it created: the same title at the same start."""
    synced = Item(connector="outlook", kind="calendar_event", external_id="outlook:study", title="COP4533 Study Session",
                  when=at(15), summary="until 5:00 PM · Library")
    plans = AgreedPlans("hemdutt")
    plans.on_agreed(agreed())
    board = dayrun(places, tmp_path, plans, memory=FakeMemory(items=[synced])).board()
    assert [b.id for b in board.blocks if "study" in b.title.lower()] == ["outlook:study"]


def test_the_block_is_rebuilt_from_the_log_and_a_scripted_day_is_left_alone(places, tmp_path):
    bus = EventBus(tmp_path / "events")
    asyncio.run(bus.publish(agreed()))
    plans = AgreedPlans("hemdutt")
    bus = EventBus(tmp_path / "events")  # a new process on the same log
    bus.subscribe("circle.event.agreed", plans.on_agreed, replay=True)
    asyncio.run(bus.replay_into())
    run = dayrun(places, tmp_path, plans)
    assert "circle:t1" in [b.id for b in run.board().blocks]
    assert "circle:t1" not in [b.id for b in run.board(scenario="demo").blocks]  # a scenario is its own day


def test_the_bootstrap_wires_agreed_plans_only_with_a_circle(tmp_path):
    from contexts.demo import circle
    from contexts.demo.seed import build_app

    alone = build_app(tmp_path / "alone")
    assert alone.dayrun.agreed is None
    joined = build_app(tmp_path / "joined", circle.config_dir(tmp_path / "config"))
    asyncio.run(joined.bus.publish(agreed(start=datetime.now(TZ).replace(microsecond=0) + timedelta(hours=3))))
    assert "circle:t1" in [b.id for b in joined.dayrun.board().blocks]
    for app in (alone, joined):
        app.projection.store.close()
