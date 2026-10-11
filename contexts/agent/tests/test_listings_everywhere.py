# OWNER'S RULE (Hemdutt): a public listing (a campus event a feed publishes) is never the person's own commitment, anywhere.
# One classifier (contexts/agent/application/schedule.py `Listings`, read from config/agent.yaml `listings:`) is handed by the
# gateway as a plain predicate to every place a calendar event becomes a commitment: the Day Run board (so the Day plan, the
# tracker's leave-by and the narrator), the Circle's free/busy answer, Habits' calendar and the leaving checklist. They treat
# a listing as they already treated the optional `nearby:` events (#284). It may still be OFFERED.
import asyncio
from datetime import timedelta
from pathlib import Path

import yaml

from contexts.agent.application import schedule
from contexts.agent.application.memory_port import FakeMemory
from contexts.agent.tests.test_planner import NOW as SATURDAY
from contexts.agent.tests.test_planner import make as make_planner
from contexts.circle.application.protocol import CircleAgent, InMemoryCircleStore, busy
from contexts.day.tests.test_day import SUNDAY, at, feed, make
from contexts.demo.story import story
from contexts.habits.tests.helpers import at as habit_at
from contexts.habits.tests.helpers import tracker
from contexts.play.domain.places import PlaceSet
from contracts.circle import Proposal
from contracts.connectors import ItemSynced
from contracts.memory import Item

LISTINGS = schedule.Listings.of(yaml.safe_load(Path("config/agent.yaml").read_text())["listings"])
TITLE = "Theatre UCF Open House"


def listing(when, until="10:45 AM", external_id="ucf-events:3900001:2026-10-11T10:00:00-04:00"):
    """A campus event known only by where its link points: no `campus:` feed prefix, no "optional" in its text."""
    return Item(connector="calendar", kind="calendar_event", external_id=external_id, title=TITLE, when=when,
                url="https://events.ucf.edu/event/3900001/theatre-ucf-open-house/", summary=f"until {until} · Student Union")


def the_day(tmp_path, listed):
    items = [Item(**e.model_dump(include={"connector", "kind", "external_id", "title", "when", "url", "summary"}))
             for e in story() if e.type == "connector.item.synced"]
    loop = make(items + [listing(at(10))], PlaceSet.load("config/places"), tmp_path)
    loop.planner.dayrun.listed = listed  # as contexts/gateway/bootstrap.py sets it
    return loop


def test_a_listing_is_never_a_fixed_block_never_gets_a_leave_by_and_the_narrator_never_says_it(tmp_path):
    loop = the_day(tmp_path / "a", None)  # before: the board took it for a class
    [plan, *_] = events = loop.planner.plan(at(8, 10))
    feed(loop, events)
    assert [(m.fixed, m.kind) for m in plan.moves if m.title == TITLE] == [(True, "class")]
    assert loop.tracker.compute(at(9, 45)).next_event.title == TITLE  # "leave by 9:50" for something that is not theirs
    assert f"{TITLE} 10 AM" in loop.narrator.job_now("morning", at(8, 15)).draft.display

    loop = the_day(tmp_path / "b", LISTINGS.listed)
    [plan, *_] = events = loop.planner.plan(at(8, 10))
    feed(loop, events)
    assert not any(m.title == TITLE for m in plan.moves)  # not a fixed block, and not chosen for them either
    board = loop.planner.dayrun.board(at(8, 10))
    assert TITLE not in [b.title for b in board.blocks] and [e.title for e in board.events] == [TITLE]  # still OFFERED, as the campus feed is
    status = loop.tracker.compute(at(9, 45))
    assert status.next_event.title == "Knight Hacks: submissions due" and status.next_block.title != TITLE  # no leave-by for a listing
    for moment in ("morning", "now"):
        said = loop.narrator.job_now(moment, at(9, 45))
        assert TITLE not in said.draft.display and "Open House" not in said.draft.spoken  # not in "your day right now" or the plan's line
    assert not any(TITLE in b["title"] for b in loop.blocks(at(8, 30)))  # nor in what the agent's rundown reads


def test_a_plan_already_agreed_with_a_listing_in_it_keeps_what_was_agreed(tmp_path):
    loop = the_day(tmp_path, None)
    [plan, *_] = events = loop.planner.plan(at(8, 10))
    feed(loop, events)  # yesterday's code: the listing is a fixed block of the plan in the log
    loop.planner.dayrun.listed = LISTINGS.listed  # the box is updated; the log replays the same plan
    again = the_day(tmp_path / "replayed", LISTINGS.listed)
    again.planner.on_plan_proposed(plan)
    again.tracker.on_change(plan)
    assert [m.title for m in again.planner.current[SUNDAY].moves] == [m.title for m in plan.moves]  # nothing rewritten
    assert any(m.title == TITLE for m in again.planner.current[SUNDAY].moves)
    assert again.tracker.compute(at(9, 45)) is not None and again.blocks(at(9, 45)) is not None  # and nothing crashes on it


def test_a_listing_never_makes_the_person_busy_for_the_circle():
    memory = FakeMemory(items=[listing(at(10))])
    assert busy(memory, at(10), at(10, 30)) is True  # before: a public open house blocked a plan with friends
    assert busy(memory, at(10), at(10, 30), LISTINGS.listed) is False
    own = Item(connector="calendar", kind="calendar_event", external_id="outlook:lecture:1", title="COP4533 lecture", when=at(10))
    assert busy(FakeMemory(items=[listing(at(10)), own]), at(10), at(10, 30), LISTINGS.listed) is True  # their own class still does
    agent = CircleAgent("me", "team", InMemoryCircleStore(), memory, clock=lambda: at(9))
    agent.listed = LISTINGS.listed  # as the gateway sets it
    proposal = Proposal(title="Lunch", start=at(10), end=at(10, 30))
    assert agent.decide(proposal).answer == "free"


def test_a_listing_is_nothing_for_habits_to_get_ready_for():
    event = ItemSynced(ts=habit_at(11, "08:00"), connector="calendar", kind="calendar_event", external_id="ucf-events:3900001:x", title=TITLE,
                       when=habit_at(11, "10:00"), url="https://events.ucf.edu/event/3900001/theatre-ucf-open-house/", summary="until 10:45 AM · Student Union")
    plain = tracker()
    plain.on_item(event)
    assert len(plain.calendar) == 1  # before
    t = tracker()
    t.cfg.is_listing = LISTINGS.listed  # as the gateway sets it
    assert t.on_item(event) == [] and t.calendar == {}
    t.on_item(ItemSynced(ts=habit_at(11, "08:01"), connector="calendar", kind="calendar_event", external_id="outlook:lecture:1",
                         title="COP4533 lecture", when=habit_at(11, "10:40")))
    assert len(t.calendar) == 1  # their own calendar is still theirs


def test_the_leaving_checklist_is_not_for_a_listing():
    planner, _ = make_planner()
    soon = listing(SATURDAY + timedelta(hours=1), external_id="ucf-events:1:x")
    planner.qa.memory = FakeMemory(items=[soon, *planner.qa.memory.items(SATURDAY - timedelta(days=1), SATURDAY + timedelta(days=2))])
    assert planner.leaving(SATURDAY).text.startswith(f"Next: {TITLE}")  # before: Listings() knows no hosts
    planner.qa.listings = LISTINGS
    assert planner.leaving(SATURDAY).text.startswith("Next: Dinner with Ilan at 6:00 PM")


def test_the_gateway_hands_the_one_classifier_to_every_guard(tmp_path):
    from fastapi.testclient import TestClient

    from contexts.gateway.app import create_app
    from contexts.gateway.bootstrap import Settings

    with TestClient(create_app(Settings(data_dir=tmp_path, token=None, connectors=False, ntfy_url=None, llm_url=None))) as c:
        atlas = c.app.state.atlas
        it = listing(at(10))
        assert atlas.qa.listings.listed(it) and atlas.qa.talk.listings is atlas.qa.listings
        assert atlas.dayrun.listed(it) and atlas.habits.base.cfg.is_listing(it)
        assert atlas.qa.talk.plan_day is not None and atlas.qa.day_plan is not None  # "yes, plan my day" and the plan's blocks
        if atlas.circle is not None:
            assert atlas.circle.listed(it)
    assert asyncio.iscoroutinefunction(atlas.qa.talk.plan_day)
