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
    chat = Item(connector="groupme", kind="message", external_id="g1", title="Ana: COP3502 exam covers recursion + linked lists",
                summary="Ana in COP4533 group chat: COP3502 exam covers recursion + linked lists", when=NOW - timedelta(hours=3))
    qa = QA(rt, FakeMemory([NIGHT], items=[EXAM, CLASS, chat]), clock=lambda: NOW, sleep=SleepSchedule(tz=NY), web=web)
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
    assert "Group chat, Sat 11:00 AM: Ana in COP4533 group chat: COP3502 exam covers recursion" in prompt
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


def test_plans_include_todays_habit_numbers():
    class Habits:
        def digest(self, now=None):
            return {"habits": [
                {"id": "meals", "name": "Meals", "unit": "meals", "direction": "at_least", "target": 2, "value": 1,
                 "met": False, "applies": True, "streak": 0},
                {"id": "sleep", "name": "Sleep", "unit": "hours", "direction": "at_least", "target": 6, "value": 4.5,
                 "met": False, "applies": True, "streak": 0},
                {"id": "desk", "name": "Focus", "unit": "hours", "direction": "at_least", "target": 2, "value": 0,
                 "met": True, "applies": False, "streak": 0},
            ]}

    planner, rt = make()
    planner.habits = Habits()
    asyncio.run(planner.plan("COP 3502 Exam 1", EXAM, NOW))
    prompt = rt.calls[-1][1][-1].content
    assert "Habit Meals: 1 meals today (target at least 2, not met yet, streak 0)" in prompt
    assert "Habit Sleep: 4.5 hours today" in prompt and "Focus" not in prompt  # not applicable today


def test_food_uses_nearby_places_with_links():
    planner, rt = make()

    async def places(kind):
        assert kind == "food"
        return [{"title": "Knightro's", "summary": "pizza, 70 m away", "url": "https://www.openstreetmap.org/node/1"}]

    planner.places = places
    planner.qa.planner = planner
    a = asyncio.run(planner.qa.ask("What should I eat?"))
    prompt = rt.calls[-1][1][-1].content
    assert "Nearby: Knightro's (pizza, 70 m away)" in prompt and "cs.ucf.edu" not in prompt  # places, not web search
    assert "https://www.openstreetmap.org/node/1" in [c.url for c in a.citations]


def test_leaving_uses_real_walking_time_when_the_place_is_known():
    from contexts.play.domain.places import PlaceSet
    from contracts.perception import ZoneEvent

    planner, _ = make()
    lecture = Item(connector="calendar", kind="calendar_event", external_id="l1", title="COP 3502 Lecture",
                   summary="until 11:15 AM \u00b7 HEC 101", when=NOW + timedelta(hours=1))
    planner.qa.memory._items = [lecture]
    planner.place_set = PlaceSet.load("config/places")
    planner.home_place = "ucf.union"
    [plan] = planner.on_zone_event(ZoneEvent(ts=NOW, camera_id="cam-door", track_id=1, label="person", zone_id="door", kind="enter"))
    # union -> hec is 380 m, 5 min on the real footpaths, plus a 5 min margin: leave by 2:50 for a 3:00 PM lecture
    assert plan.text.startswith("Next: COP 3502 Lecture at 3:00 PM; leave by 2:50 PM (5 min walk to L3Harris Engineering Center).")


def test_locate_finds_places_by_alias_or_name():
    from contexts.play.domain.places import PlaceSet

    ps = PlaceSet.load("config/places")
    assert ps.locate("until 11:15 AM \u00b7 HEC 101") == "ucf.hec"
    assert ps.locate("Study group at the John C. Hitt Library") == "ucf.library"
    assert ps.locate("Lunch at Chipotle") is None


def _desk(id, hours_ago, *captions):
    return Episode(id=id, camera_id="cam-desk", start=NOW - timedelta(hours=hours_ago), labels=["person"],
                   captions=list(captions), keyframe_refs=[f"frames:cam-desk:{id}"])


def test_food_cites_meals_not_mugs_cans_and_bottles():
    """The answer cited the package, the keys and a water bottle, and pushed the bagels out of the last six."""
    planner, rt = make()
    planner.qa.memory._episodes += [
        _desk("bagels", 6.5, "Two people eat bagels for breakfast at the table and share a coffee.",
              "On the desk: bagel (on a plate), coffee mug."),  # one episode, cited once
        _desk("package", 6, "A person opens the cardboard package on the table.",
              "On the desk: package (open, in the middle of the table), keys (next to the coffee mug, left of the closed laptop)."),
        _desk("cans", 5.5, "A person collects three empty cans from the desk."),
        _desk("bins", 5.4, "A person drops two cans into the recycling bin and one into the trash.", "On the recycling: can (in the recycling bin)."),
        _desk("bottle", 5, "A person holds a water bottle up to the camera, then drops it in the recycling bin."),
        _desk("bowl", 4.5, "A person drops a food-stained takeout bowl in the recycling bin."),
        _desk("candle", 4, "A person lights a candle next to the canvas bag."),
        _desk("keys", 3, "On the desk: keys (left of the laptop, behind the coffee mug), laptop, phone."),
        _desk("pizza", 2, "Three people eat pizza at the table; an open pizza box sits between the laptops."),
        _desk("drink", 1, "A person opens an energy drink can and sets it next to the laptop."),
    ]
    planner.qa.planner = planner
    a = asyncio.run(planner.qa.ask("What should I eat?"))
    seen = [c for c in a.citations if c.episode_id]
    assert [c.episode_id for c in seen] == ["bagels", "pizza", "drink"]  # time order, each episode once
    assert [c.frame_ref for c in seen] == ["frames:cam-desk:bagels", "frames:cam-desk:pizza", "frames:cam-desk:drink"]
    prompt = rt.calls[-1][1][-1].content
    assert "Seen today at Sat 7:30 AM: Two people eat bagels for breakfast" in prompt
    assert "On the desk" not in prompt and "water bottle" not in prompt and "No meals seen" not in prompt


def test_food_keeps_the_meals_when_there_is_more_than_fits():
    from contexts.agent.application.planner import MAX_MEALS, meals_seen

    snacks = [_desk(f"chips{n}", 5 - n / 2, "A bag of chips lies next to the keyboard.") for n in range(MAX_MEALS + 2)]
    breakfast = _desk("bagels", 6, "Two people eat bagels for breakfast.")
    kept = meals_seen([breakfast, *snacks])
    assert len(kept) == MAX_MEALS and kept[0][0].id == "bagels"  # the real meal stays; the oldest snacks go
    assert [ep.start for ep, _ in kept] == sorted(ep.start for ep, _ in kept)


def test_food_never_says_no_meals_beside_habits_meal_count():
    """Round 2: "No meals seen at the desk today" next to "Habit Meals: 2 meals today"."""
    class Habits:
        def __init__(self, meals):
            self.meals = meals

        def digest(self, now=None):
            return {"habits": [{"id": "meals", "name": "Meals", "unit": "meals", "direction": "at_least", "target": 2,
                                "value": self.meals, "met": self.meals >= 2, "applies": True, "streak": 0}]}

    for meals, says_none in ((2, False), (0, True)):
        planner, rt = make()
        planner.habits = Habits(meals)
        asyncio.run(planner.food("What should I eat?"))
        prompt = rt.calls[-1][1][-1].content
        assert f"Habit Meals: {meals} meals today" in prompt
        assert ("No meals seen at the desk today." in prompt) == says_none


CONCERT = Item(connector="calendar", kind="calendar_event", external_id="campus:concert", title="Free concert on the lawn",
               when=NOW + timedelta(hours=6), url="https://events.test/concert", summary="until 10:00 PM · Memory Mall · free")
NOTICE = Item(connector="canvas", kind="assignment", external_id="x2", when=NOW + timedelta(hours=9),
              title="GatorEvals – Fall 2026 Mid-Term Evaluation Voluntary Opt-In", url="https://canvas.test/evals")


def test_only_graded_work_is_a_crunch_item():
    """A chat message with "due", a "submissions due" calendar entry and an opt-in notice all got study blocks."""
    soon = NOW + timedelta(hours=3)
    chat = Item(connector="groupme", kind="message", external_id="g2", title="Maya: anyone started the milestone? due tuesday", when=soon)
    submit = Item(connector="calendar", kind="calendar_event", external_id="c2", title="Knight Hacks: submissions due", when=soon)
    midterm = Item(connector="calendar", kind="calendar_event", external_id="c3", title="COP 3502 Midterm", when=soon)
    quiz = Item(connector="canvas", kind="assignment", external_id="x3", title="Quiz 5 - Greedy Algorithms [COP4533]", when=soon)
    survey = quiz.model_copy(update={"title": "Course survey (voluntary)"})
    assert [is_crunch(i, NOW) for i in (chat, submit, NOTICE, survey)] == [False] * 4
    assert is_crunch(midterm, NOW) and is_crunch(quiz, NOW) and is_crunch(EXAM, NOW)
    planner, rt = make()
    assert asyncio.run(planner.on_item(ItemSynced(ts=NOW, **NOTICE.model_dump()))) == [] and not rt.calls  # no plan, no blocks


def test_tonight_question_answers_from_this_evening_even_with_a_deadline():
    """ "Where should I go tonight?" planned around the next "due" item and never listed tonight's events."""
    from contexts.circle.application.presence import PresenceBoard
    from contracts.circle import PlaceShared

    spec = ConnectorSpec(name="calendar", module="x", writes=["create_event"])
    for question in ("Where should I go tonight?", "What should I do tonight?", "what should I do this evening"):
        planner, rt = make(ConnectorHub([spec], factory=lambda s: FakeConnector()))
        tomorrow = CLASS.model_copy(update={"external_id": "c9", "title": "Brunch", "when": NOW + timedelta(hours=21)})
        planner.qa.memory._items += [CONCERT, NOTICE, tomorrow]
        planner.presence = PresenceBoard("team", "dom", ["Library"], opted_in=True)
        planner.presence.on_shared(PlaceShared(ts=NOW, circle_id="team", member_id="hemdutt", place="Library", source="phone",
                                               expires_at=NOW + timedelta(hours=2)))
        plan = asyncio.run(planner.answer(question))
        assert plan.subject == question and plan.blocks == []  # the evening, not study blocks for the exam
        lines = rt.calls[-1][1][-1].content.splitlines()
        assert lines[1:6] == [
            "- Coming up: Dinner with Ilan at Sat 6:00 PM",
            "- Coming up: Free concert on the lawn at Sat 8:00 PM (until 10:00 PM · Memory Mall · free)",
            "- hemdutt is at Library (shared until 4:00 PM)",
            "- Deadline coming up: COP 3502 Exam 1 at Sun 10:00 AM",  # the real deadline: one line, after the evening
            "- Usual bedtime 08:00 PM",
        ]
        assert "GatorEvals" not in rt.calls[-1][1][-1].content and "Brunch" not in rt.calls[-1][1][-1].content
        assert [(c.title, c.url) for c in plan.citations] == [
            ("Dinner with Ilan", None), ("Free concert on the lawn", "https://events.test/concert"),
            ("hemdutt at Library", None), ("COP 3502 Exam 1", "https://canvas.test/exam1")]

    planner, rt = make()  # studying for something tonight is still the exam plan
    plan = asyncio.run(planner.answer("Help me prep for my exam tonight"))
    assert plan.subject == "COP 3502 Exam 1" and plan.blocks


def test_tonight_with_nothing_on_says_so_and_started_events_still_count():
    planner, rt = make()
    planner.qa.memory._items = [CLASS.model_copy(update={"when": NOW - timedelta(minutes=30)}),
                                CLASS.model_copy(update={"external_id": "old", "title": "Lunch", "when": NOW - timedelta(hours=3)})]
    asyncio.run(planner.answer("Where should I go tonight?"))
    assert "- Coming up: Dinner with Ilan at Sat 1:30 PM, already started" in rt.calls[-1][1][-1].content
    assert "Lunch" not in rt.calls[-1][1][-1].content
    planner, rt = make()
    planner.qa.memory._items = []
    plan = asyncio.run(planner.answer("Where should I go tonight?"))
    assert "- Nothing on the calendar between now and Sat 10:00 PM." in rt.calls[-1][1][-1].content and plan.citations == []


def test_the_evening_runs_to_bedtime_and_at_least_eight_hours():
    from datetime import time

    from contexts.agent.application.planner import evening_end

    early, late = SleepSchedule(tz=NY), SleepSchedule(bedtime=time(1), wake=time(9), tz=NY)
    morning = datetime(2026, 10, 10, 10, 30, tzinfo=NY)
    assert evening_end(morning, early) == datetime(2026, 10, 10, 20, tzinfo=NY)  # until bedtime
    assert evening_end(NOW, early) == NOW + timedelta(hours=8)  # 2 PM: bedtime is 6 hours off, the evening is not over at 8
    assert evening_end(NOW.replace(hour=21), early) == NOW.replace(hour=21) + timedelta(hours=8)  # already past bedtime
    assert evening_end(NOW, late) == datetime(2026, 10, 11, 1, tzinfo=NY)  # a bedtime after midnight is tonight's
