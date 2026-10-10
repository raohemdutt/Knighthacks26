import asyncio
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from contexts.agent.application.memory_port import FakeMemory
from contexts.agent.application.planner import Planner, free_blocks, is_crunch
from contexts.agent.application.qa import QA, SleepSchedule
from contexts.agent.application.web_port import FakeWeb
from contexts.connectors.application.hub import ConnectorHub
from contexts.connectors.domain.spec import ConnectorSpec
from contexts.connectors.fakes.fake import FakeConnector
from contracts.agent import ActionProposed, PlanCreated
from contracts.connectors import ItemSynced
from contracts.memory import Episode, Item
from contracts.types import ChatResult
from runtime import FakeRuntime

NY = ZoneInfo("America/New_York")
NOW = datetime(2026, 10, 10, 14, 0, tzinfo=NY)  # Saturday 2 PM
EXAM = Item(connector="canvas", kind="assignment", external_id="x1", title="COP 3502 Exam 1",
            when=NOW + timedelta(hours=20), url="https://canvas.test/exam1")
CLASS = Item(connector="calendar", kind="calendar_event", external_id="c1", title="Dinner with Ilan",
             when=NOW + timedelta(hours=4), url=None)
NIGHT = Episode(id="n1", camera_id="cam-desk", start=datetime(2026, 10, 10, 2, 30, tzinfo=NY), labels=["person"], captions=["A person types at the desk."])


def make(hub=None):
    web = FakeWeb(results=[{"title": "COP 3502 Exam Info", "url": "https://cs.ucf.edu/exam/", "summary": "Exam 1: recursion, linked lists"}],
                  posts=[{"title": "Exam 1 tips", "url": "https://reddit.com/r/ucf/1"}])
    rt = FakeRuntime(chat_replies=[ChatResult(content="- Study recursion 3-5 PM\n- Sleep by 11 PM", model_version="v1")])
    qa = QA(rt, FakeMemory([NIGHT], items=[EXAM, CLASS]), clock=lambda: NOW, sleep=SleepSchedule(tz=NY), web=web)
    return Planner(qa, hub), rt


def test_crunch_detection():
    assert is_crunch(EXAM, NOW)
    assert not is_crunch(CLASS, NOW)
    assert not is_crunch(EXAM.model_copy(update={"when": NOW + timedelta(days=3)}), NOW)


def test_free_blocks_skip_sleep_and_busy_time():
    planner, _ = make()
    blocks = free_blocks(NOW, EXAM.when, [CLASS], planner.qa)
    # 3-5 PM free; dinner at 6 blocks 6-8; 8 PM to 8 AM is sleep; 8-10 AM fits before the 10 AM exam
    assert [(s.hour, e.hour) for s, e in blocks] == [(15, 17), (8, 10)]


def test_crunch_item_triggers_a_cited_plan_and_a_calendar_proposal():
    spec = ConnectorSpec(name="calendar", module="x", writes=["create_event"])
    hub = ConnectorHub([spec], factory=lambda s: FakeConnector())
    planner, rt = make(hub)
    synced = ItemSynced(ts=NOW, **EXAM.model_dump())
    plan, proposal = asyncio.run(planner.on_item(synced))
    assert isinstance(plan, PlanCreated) and plan.subject == "COP 3502 Exam 1"
    assert plan.text.startswith("- Study recursion")
    urls = [c.url for c in plan.citations]
    assert urls[:1] == ["https://canvas.test/exam1"] and "https://cs.ucf.edu/exam/" in urls and "https://reddit.com/r/ucf/1" in urls
    prompt = rt.calls[-1][1][-1].content
    assert "activity at the desk 1 times, last at Sat 2:30 AM" in prompt and "Free slots:" in prompt
    assert isinstance(proposal, ActionProposed) and proposal.target == "calendar.create_event"
    assert len(proposal.params["events"]) == 2
    assert asyncio.run(planner.on_item(synced)) == []  # once per item


def test_plan_question_without_calendar_write_just_plans():
    planner, _ = make()
    plan = asyncio.run(planner.answer("Help me prep for my exam"))
    assert plan.subject == "COP 3502 Exam 1" and plan.blocks


def test_qa_routes_plan_questions_to_the_planner():
    planner, _ = make()
    planner.qa.planner = planner
    a = asyncio.run(planner.qa.ask("What should I do tonight to prepare?"))
    assert a.text.startswith("- Study recursion") and a.citations


def test_leaving_checklist_at_the_door():
    from contracts.memory import Sighting
    from contracts.perception import ZoneEvent

    planner, _ = make()
    planner.qa.memory._sightings["keys"].append(
        Sighting(label="keys", camera_id="cam-desk", zone_id="desk", ts=NOW - timedelta(minutes=30), frame_ref="frames:cam-desk:k", position="by the mug"))
    enter = ZoneEvent(ts=NOW, camera_id="cam-door", track_id=1, label="person", zone_id="door", kind="enter")
    [plan] = planner.on_zone_event(enter)
    assert plan.subject == "Leaving"
    assert plan.text == "Next: Dinner with Ilan at 6:00 PM; leave by 5:40 PM. Keys: last seen in the desk zone, by the mug at 1:30 PM."
    assert plan.citations[1].frame_ref == "frames:cam-desk:k"
    assert planner.on_zone_event(enter.model_copy(update={"ts": NOW + timedelta(minutes=5)})) == []  # cooldown
    assert planner.on_zone_event(enter.model_copy(update={"label": "package"})) == []


def test_food_question_uses_what_was_eaten_today():
    planner, rt = make()
    planner.qa.memory._episodes.append(
        Episode(id="m1", camera_id="cam-desk", start=NOW - timedelta(hours=2), labels=["person"], captions=["A person eats pizza at the desk."]))
    planner.qa.planner = planner
    a = asyncio.run(planner.qa.ask("What should I eat for dinner?"))
    prompt = rt.calls[-1][1][-1].content
    assert "eats pizza" in prompt and "cs.ucf.edu" in prompt  # FakeWeb result stands in for nearby places
    assert a.citations[0].episode_id == "m1"
