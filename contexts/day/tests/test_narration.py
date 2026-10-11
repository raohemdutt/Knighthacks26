"""The narrator (D3): spoken numbers and times, the template engine, the grounding check, the model engine's
fallbacks, the cache and the rate limit, replay, the API and the push."""

import asyncio
import json
import threading
import time as clocktime
from datetime import date, datetime, timedelta

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from bus import EventBus
from contexts.agent.application.memory_port import FakeMemory
from contexts.day.api import day_router
from contexts.day.application.loop import DayLoop, day_phone
from contexts.day.application import narrator as narrator_module
from contexts.day.application.narrator import MIN_DRAFT_CHARS, NarrationConfig
from contexts.day.domain import narration as n
from contexts.day.domain.narration import Words, check, render, shape
from contexts.day.tests.test_day import CLOCK, TZ, FakeHabits, at, hub, items, never, places  # noqa: F401 (fixtures)
from contexts.connectors.application.hub import ConnectorHub
from contexts.play.dayrun.service import DayRun
from contracts.agent import AnswerGenerated
from contracts.day import DayNarrated, DayPlanProposed, DayStatusChanged, PlanMove
from contracts.experience import QuestionAsked
from contracts.types import ChatResult
from contracts.workspace import ApprovalRequested
from runtime.fake import FakeRuntime

# --- numbers, durations and times the way people say them ---


@pytest.mark.parametrize("value, said", [(0, "zero"), (8, "eight"), (13, "thirteen"), (20, "twenty"), (25, "twenty-five"),
                                         (90, "ninety"), (100, "a hundred"), (120, "a hundred and twenty"), (345, "three hundred and forty-five"),
                                         (1200, "1200")])
def test_numbers_in_words(value, said):
    assert n.say_number(value) == said


@pytest.mark.parametrize("minutes, said", [(1, "a minute"), (7, "seven minutes"), (10, "ten minutes"), (12, "about ten minutes"),
                                           (20, "twenty minutes"), (28, "about half an hour"), (30, "half an hour"), (45, "forty-five minutes"),
                                           (60, "an hour"), (62, "about an hour"), (75, "an hour and a quarter"), (90, "ninety minutes"),
                                           (100, "about an hour and three quarters"), (120, "two hours"), (150, "two and a half hours"),
                                           (171, "about three hours"), (300, "five hours"), (0, "no time")])
def test_durations_are_rounded_the_way_people_say_them(minutes, said):
    assert n.say_minutes(minutes) == said


@pytest.mark.parametrize("hours, said", [(4.1, "about four hours"), (5.4, "about five and a half hours"), (6.0, "six hours"), (7.8, "about eight hours"),
                                         (1.0, "an hour"), (1.4, "about an hour and a half"), (0.5, "half an hour"), (2.22, "about two hours")])
def test_hours_of_sleep(hours, said):
    assert n.say_hours(hours) == said


@pytest.mark.parametrize("clock, said", [("15:30", "half past three"), ("15:15", "quarter past three"), ("15:45", "quarter to four"),
                                         ("12:50", "ten to one"), ("09:20", "twenty past nine"), ("09:00", "nine o'clock"), ("12:00", "noon"),
                                         ("00:00", "midnight"), ("14:40", "twenty to three"), ("08:31", "half past eight"), ("10:51", "ten to eleven"),
                                         ("11:58", "noon"), ("23:40", "twenty to midnight"), ("11:35", "twenty-five to noon"), ("00:10", "ten past twelve")])
def test_clock_times_as_said(clock, said):
    assert n.say_time(clock) == said


def test_a_time_to_leave_by_never_rounds_later_and_the_screen_keeps_it_exact():
    assert n.say_time("14:28") == "half past two" and n.say_time("14:28", down=True) == "twenty-five past two"
    assert n.say_time(datetime(2026, 10, 11, 14, 28, tzinfo=TZ), down=True) == "twenty-five past two"
    assert n.clock("14:28") == "2:28 PM" and n.clock("09:00") == "9 AM" and n.clock("00:05") == "12:05 AM"


def test_days_shares_and_names():
    today = date(2026, 10, 11)
    assert [n.say_day(today + timedelta(days=d), today) for d in (0, 1, 2, -1)] == ["today", "tomorrow", "Tuesday", "yesterday"]
    assert [n.say_share(p) for p in (0, 25, 33, 50, 62, 66, 75, 95, 100, 85)] == [
        "almost none", "about a quarter", "about a third", "about half", "about two thirds", "about two thirds", "about three quarters",
        "nearly all", "all", "about eighty-five percent"]
    assert n.short_title("Project - Milestone1 Fall2026 [COP4533]") == "Project Milestone 1"
    assert n.short_title("Knight Hacks: mentor check-in") == "mentor check-in"
    assert n.short_title("Knight Hacks: dinner") == "Knight Hacks dinner"  # one generic word would be mistaken for a meal
    assert n.short_title("COP4533 study session") == "COP4533 study session"
    assert n.short_place("Engineering II (Knight Hacks)") == "Engineering II" and n.short_place("Desk") == "the desk"
    assert n.short_place("John C. Hitt Library") == "John C. Hitt Library" and n.short_place(None) is None
    assert n.the("mentor check-in") == "the mentor check-in" and n.the("COP4533 study session") == "COP4533 study session"


# --- the shape of a plan: a few beats, not every block ---


def move(i: int, kind: str, start: datetime, minutes: int, title: str | None = None, fixed: bool = False, place: str = "Desk",
         deadline: str | None = None) -> PlanMove:
    title = title or {"study": f"Study at {place}", "eat": f"Eat at {place}", "rest": f"Rest at {place}", "sleep": "Sleep"}[kind]
    return PlanMove(id=f"f{i}" if fixed else f"m{i}", kind=kind, title=title, start=start, end=start + timedelta(minutes=minutes),
                    minutes=minutes, place=f"apartment.{place.lower()}", place_name=place, fixed=fixed, deadline=deadline)


def sunday() -> list[PlanMove]:
    """A light Sunday: a nap, two short study blocks with a breather, the mentor check-in, an afternoon, the night."""
    return [
        move(0, "sleep", at(8, 30), 60),
        move(1, "study", at(9, 40), 25, deadline="hw"), move(2, "rest", at(10, 5), 20), move(3, "study", at(10, 25), 25, deadline="hw"),
        move(4, "eat", at(12, 0), 30, place="Kitchen"),
        move(5, "class", at(12, 50), 20, title="Knight Hacks: mentor check-in", fixed=True, place="Engineering II (Knight Hacks)"),
        move(6, "study", at(14, 0), 90, deadline="hw"),
        move(7, "class", at(15, 30), 120, title="Knight Hacks: judging and demos", fixed=True, place="Engineering II (Knight Hacks)"),
        move(8, "eat", at(18, 0), 30, place="Kitchen"),
        move(9, "sleep", at(23, 0), 510),
        move(10, "study", at(8, 30, day=12), 25), move(11, "rest", at(8, 55, day=12), 20), move(12, "study", at(9, 15, day=12), 25),
    ]


def test_consecutive_blocks_collapse_and_only_fixed_events_are_named():
    s = shape(sunday(), at(8, 10), TZ)
    assert s["before"] == [{"kind": "nap", "count": 1, "min": 60, "at": "08:30"},
                           {"kind": "study", "count": 2, "min": 50, "at": "09:40"},  # the 20-minute rest between them is a breather, not a beat
                           {"kind": "meal", "count": 1, "min": 30, "at": "12:00", "place": "the kitchen"}]
    assert s["first"] == {"title": "mentor check-in", "at": "12:50", "place": "Engineering II"}
    assert s["later"] == [{"title": "judging and demos", "at": "15:30", "place": "Engineering II"}] and s["main_later"] == s["later"][0]
    assert s["after"] == [{"kind": "study", "count": 1, "min": 90, "at": "14:00", "place": "the desk"},
                          {"kind": "meal", "count": 1, "min": 30, "at": "18:00", "place": "the kitchen"}]
    assert s["bed"] == "23:00" and s["tomorrow"]["count"] == 2 and s["tomorrow"]["min"] == 50
    # later in the day, what has ended is gone and the next fixed event is the anchor
    late = shape(sunday(), at(13, 30), TZ)
    assert [b["kind"] for b in late["before"]] == ["study"] and late["first"]["title"] == "judging and demos" and "later" not in late
    # a re-plan's own subject is not told again
    assert shape(sunday(), at(8, 10), TZ, skip="f5")["first"]["title"] == "judging and demos"


def morning_facts(**extra) -> dict:
    f = {"moment": "morning", "greeting": "morning", "day": "Sunday", "sleep_hours": 4.1, **shape(sunday(), at(8, 10), TZ)}
    f["first"] |= {"walk_min": 9, "leave_by": "12:41"}
    f["deadline"] = {"title": "Milestone 1", "due_day": "Tuesday", "due": "23:59", "left_min": 300, "planned_min": 140}
    f |= {"blocks": 8, "ask": "calendar"}
    return f | extra


def test_the_morning_brief_leads_with_sleep_groups_the_day_and_ends_with_the_ask():
    text = render(morning_facts())
    assert text.spoken == (
        "Morning. You only got about four hours, so I've kept the start gentle: a nap first, then two short study blocks before the mentor "
        "check-in at ten to one. Leave by twenty to one for that; it's a nine-minute walk. Milestone 1 is due Tuesday with five hours of "
        "work left; this plan puts in about two and a half hours. Want me to put this on your calendar?")
    assert text.display == (
        "You slept 4.1 h, so the day starts gently. A nap at 8:30 AM, then two 25-minute study blocks at 9:40 AM and lunch at 12 PM before "
        "the mentor check-in at 12:50 PM (Engineering II). 9 min walk: leave by 12:41 PM. Later: judging and demos 3:30 PM. Bed at 11 PM. "
        "Tomorrow morning: two 25-minute study blocks. Milestone 1 is due Tue 11:59 PM, about 300 focused minutes left; this plan puts in 140. "
        "Approve to put 8 blocks on your calendar.")
    # for the ear: no digits, no list, a handful of sentences; every block's clock time is not recited
    assert not any(ch.isdigit() for ch in text.spoken.replace("Milestone 1", "")) and "8:30" not in text.spoken
    assert check(text.spoken, morning_facts(), text.spoken) == []


def test_the_morning_brief_bends_to_what_it_knows():
    rested = render(morning_facts(sleep_hours=7.8, ask=None))
    assert rested.spoken.startswith("Morning. You got about eight hours of sleep: a nap first") and "Want me" not in rested.spoken
    unknown = morning_facts()
    del unknown["sleep_hours"], unknown["deadline"]
    assert render(unknown).spoken.startswith("Morning. Here is the plan: a nap first") or render(unknown).spoken.startswith("Morning. Today looks like this: a nap first")
    mine = render(morning_facts(by="person"))
    assert "Your version of the day: a nap first" in mine.spoken and mine.spoken.endswith("Want me to put your version on your calendar?")
    no_calendar = render(morning_facts(ask="plan"))
    assert no_calendar.spoken.endswith("Want me to keep this as today's plan?") and "nothing is written" in no_calendar.display
    covered = render(morning_facts(deadline={"title": "Milestone 1", "due_day": "tomorrow", "due": "17:00", "left_min": 90, "planned_min": 140}))
    assert "Milestone 1 is due tomorrow, and this plan covers the ninety minutes it still needs." in covered.spoken
    # nothing fixed: the beats alone; one beat before a fixed event needs no "first"
    free = {"moment": "morning", "greeting": "afternoon", "day": "Sunday", "before": [{"kind": "study", "count": 1, "min": 90, "at": "14:00"}]}
    assert render(free).spoken in ("Afternoon. Here is the plan: a long study block first.", "Afternoon. Today looks like this: a long study block first.")
    one = {"moment": "morning", "greeting": "morning", "day": "Sunday", "before": [{"kind": "rest", "count": 1, "min": 20, "at": "09:00"}],
           "first": {"title": "Judging", "at": "15:30"}, "main_later": {"title": "team dinner", "at": "19:00"}}
    assert render(one).spoken == "Morning. Today looks like this: a break before Judging at half past three, and the team dinner this evening." or \
        render(one).spoken == "Morning. Here is the plan: a break before Judging at half past three, and the team dinner this evening."


def test_now_says_where_you_are_against_the_plan_and_when_to_move():
    nxt = {"title": "judging and demos", "at": "15:30", "place": "Engineering II", "walk_min": 9, "leave_by": "15:21", "leave_in_min": 10}
    behind = render({"moment": "now", "risk": "ok", "block": "your study block", "until": "15:10", "should_be": "the desk", "seen_at": "the front door",
                     "drift_min": 20, "next": nxt})
    assert behind.spoken == ("You're about twenty minutes behind on your study block: the plan has you at the desk and I last saw you at the front door. "
                             "The judging and demos is at half past three; leave by twenty past three.")
    assert behind.display == ("About 20 min behind on your study block until 3:10 PM; last seen at the front door. "
                              "The judging and demos is at 3:30 PM at Engineering II (9 min walk, leave by 3:21 PM).")
    on_plan = render({"moment": "now", "risk": "ok", "block": "lunch", "until": "13:00", "next": nxt})
    assert on_plan.spoken.startswith("You're on plan: lunch until one o'clock.")
    leave = render({"moment": "now", "risk": "leave_now", "seen_at": "the desk", "next": nxt})
    assert leave.spoken == "Time to go. The judging and demos starts at half past three at Engineering II, a nine-minute walk from here; leave in ten minutes."
    assert render({"moment": "now", "risk": "leave_now", "next": nxt | {"leave_in_min": 0}}).spoken.endswith("leave now.")
    late = render({"moment": "now", "risk": "late", "seen_at": "the desk", "next": nxt})
    assert late.spoken == "The judging and demos started at half past three and you're still at the desk. Head to Engineering II now; it's a nine-minute walk."
    missed = render({"moment": "now", "risk": "infeasible", "seen_at": "the desk", "next": nxt})
    assert missed.spoken.startswith("You won't make the judging and demos at half past three from the desk; it's a nine-minute walk.")
    free = render({"moment": "now", "risk": "ok", "then": {"title": "the judging and demos", "at": "15:30"}, "next": nxt})
    assert free.spoken == "You're free until the judging and demos at half past three. Leave by twenty past three."  # one event, said once
    assert render({"moment": "now", "risk": "ok"}).spoken == "Nothing is planned right now."


def test_replan_coach_evening_and_the_asks():
    replan = render({"moment": "replan", "seen_at": "the desk", "before": [{"kind": "nap", "count": 1, "min": 60, "at": "15:10"}],
                     "first": {"title": "judging and demos", "at": "17:20", "place": "Engineering II"}, "blocks": 6, "ask": "calendar",
                     "for": {"title": "mentor check-in", "at": "14:40", "place": "Engineering II", "walk_min": 13, "kept": True}})
    assert replan.spoken == ("You won't make the start of the mentor check-in at twenty to three from the desk; it's a thirteen-minute walk. Head there now "
                             "and catch what's left. After that I'd do it this way: a nap before the judging and demos at twenty past five. "
                             "Want me to switch to that?")
    assert "Approve to replace today's blocks with these 6." in replan.display
    asked = render({"moment": "replan", "asked": True, "seen_at": "the desk", "before": [{"kind": "study", "count": 2, "min": 50, "at": "15:10"}]})
    assert asked.spoken == "Here's a fresh plan from the desk: two short study blocks first."

    coach = render({"moment": "coach", "title": "milestone 1 writeup", "planned_min": 50, "desk_min": 15, "breaks": 2,
                    "long_break": {"min": 17, "trigger": "chat"}, "longest_min": 10, "tip": "Hold chat notifications until the break", "block_min": 25})
    assert coach.spoken == ("That one got away from you a little: fifteen of fifty minutes at the desk. Two breaks, and the long one started right after the "
                            "group chat lit up. Next time, hold chat notifications until the break. I'll plan twenty-five-minute blocks from here.")
    assert coach.display.startswith("Milestone 1 writeup: 15 of 50 minutes at the desk. 2 breaks; the longest was 17 min")
    good = render({"moment": "coach", "title": "Study at Desk", "planned_min": 50, "desk_min": 47, "breaks": 0, "next": {"min": 25, "at": "16:00"}})
    assert good.spoken == "Nice work: forty-seven of fifty minutes at the desk. No breaks. Your next study block is twenty-five minutes at four o'clock."

    evening = render({"moment": "evening", "score": 62, "stars": 1, "points": 64, "plan_stars": 2, "followed": 3, "slipped": 1,
                      "win": {"kind": "stretch", "min": 40, "title": "afternoon study block"}, "tomorrow": "Start the first study block earlier tomorrow, before anything can push it."})
    assert evening.spoken == ("That's the day. You followed about two thirds of the plan and earned one star. The win: an unbroken forty minutes in the "
                              "afternoon study block. For tomorrow: start the first study block earlier, before anything can push it.")
    assert evening.display.startswith("You followed 62% of what could be measured: 1 star, 64 points (the plan would have earned 2).")
    unmeasured = render({"moment": "evening", "score": None, "stars": 0})
    assert unmeasured.spoken.startswith("That's the day. I couldn't measure enough of it to say how it went.")

    add = render({"moment": "approval", "kind": "calendar", "count": 8, "day": "tomorrow", "replace": False, "writes": True})
    assert add.spoken == "I'd like to add eight blocks to your calendar for tomorrow. Okay?"
    assert render({"moment": "approval", "kind": "calendar", "count": 17, "day": "today", "replace": True, "writes": True}).spoken == \
        "I'd like to swap today's calendar blocks for seventeen new ones. Okay?"
    assert render({"moment": "approval", "kind": "calendar", "count": 5, "day": "today", "writes": False}).spoken == "I'd like to make these five blocks today's plan. Okay?"
    assert render({"moment": "approval", "kind": "league_dare", "title": "No energy drinks", "days": 3, "who": "Maya, Dom"}).spoken == \
        "I'd like to start a League dare: no energy drinks for three days with Maya, Dom. Okay?"
    assert render({"moment": "approval", "kind": "league_stake", "text": "Loser buys coffee"}).spoken == "I'd like to set this week's League stake: loser buys coffee. Okay?"


def test_another_profiles_words():
    shift = Words(activity="work", workplace="station", event="shift", deadline="handover")
    text = render(morning_facts(), shift)
    assert "two short work blocks" in text.spoken and "study" not in text.spoken and "two 25-minute work blocks" in text.display
    coach = render({"moment": "coach", "title": "Work at Station", "planned_min": 60, "desk_min": 40, "breaks": 0, "next": {"min": 30, "at": "20:00"}}, shift)
    assert coach.spoken == "Decent block: forty of an hour at the station. No breaks. Your next work block is half an hour at eight o'clock."
    tone = n.tone_line(shift, "A night-shift worker whose day starts in the afternoon.", "Keep it dry.")
    assert tone.startswith("Who you are talking to: A night-shift worker whose day starts in the afternoon. This person's words: \"work\"")
    assert "\"station\"" in tone and "\"shift\"" in tone and "\"handover\"" in tone and tone.endswith("Keep it dry.")
    assert n.facts_hash(morning_facts(), tone) != n.facts_hash(morning_facts(), n.tone_line(Words()))  # another tone, another narration


# --- the grounding check ---

FAITHFUL = ("Good morning. You only got about four hours, so I've kept the start gentle: a nap first, then two short study blocks before "
            "the mentor check-in at ten to one. Leave by twenty to one, it's a nine-minute walk. Milestone 1 is due Tuesday with five hours of "
            "work left, and this plan puts in about two and a half hours. Shall I put this on your calendar?")


def problems(text: str, facts: dict | None = None) -> list[str]:
    facts = facts or morning_facts()
    return check(text, facts, render(facts).spoken)


def test_a_faithful_rewrite_is_accepted():
    assert problems(FAITHFUL) == []
    # digits and the screen's clock forms of the same facts are fine too, and so is the exact leave-by
    assert problems(FAITHFUL.replace("ten to one", "12:50").replace("twenty to one", "12:41 PM").replace("nine-minute", "9-minute")) == []
    assert problems(FAITHFUL.replace("about four hours", "4.1 hours").replace("five hours", "300 minutes")) == []


@pytest.mark.parametrize("change, why", [
    (("ten to one", "half past one"), "a time that is not in the facts"),  # an invented time
    (("ten to one", "1:15 PM"), "a time that is not in the facts"),
    (("ten to one", "one o'clock"), "a time that is not in the facts"),
    (("about four hours", "about six hours"), "a number that is not in the facts"),  # a changed number
    (("two short study blocks", "three short study blocks"), "a number that is not in the facts"),
    (("five hours", "45 minutes"), "a number that is not in the facts"),
    (("a nine-minute walk", "half an hour away"), "a number that is not in the facts"),
    (("Milestone 1 is due Tuesday", "Milestone 1 is due Wednesday"), "a claim or an event that is not in the facts"),
    (("a nap first", "a quiz first"), "a claim or an event that is not in the facts"),  # an invented event
    (("the mentor check-in", "the Hackathon kickoff"), "a name that is not in the facts"),
    (("Milestone 1 is due", "Milestone 1 is overdue and was due"), "a claim or an event that is not in the facts"),
    ((" Shall I put this on your calendar?", ""), "dropped the closing question"),
])
def test_an_unfaithful_rewrite_is_rejected(change, why):
    found = problems(FAITHFUL.replace(*change))
    assert found and any(p.startswith(why) for p in found), found


def test_lists_markdown_and_length_are_rejected():
    assert any("list or markdown" in p for p in problems("Morning.\n- a nap first\n- two short study blocks\nWant me to put this on your calendar?"))
    assert any("list or markdown" in p for p in problems(FAITHFUL.replace("a nap first", "**a nap** first")))
    assert any("list or markdown" in p for p in problems("Morning. 1. a nap first 2. two short study blocks before the mentor check-in. Okay?"))
    long = FAITHFUL.replace("Shall I", "It should be a good day if you keep to it, and I'll be here if anything moves, so don't worry about the details. " * 3 + "Shall I")
    assert any(p.startswith("too long") for p in problems(long))
    many = "You're on plan. Lunch. Until one. Then go. Leave by twenty past three."
    now = {"moment": "now", "risk": "ok", "block": "lunch", "until": "13:00",
           "next": {"title": "judging and demos", "at": "15:30", "walk_min": 9, "leave_by": "15:21", "leave_in_min": 10}}
    assert any(p.startswith("too many sentences") for p in problems(many, now))
    assert problems("", now) == ["empty"] and any("too short" in p for p in problems("Lunch now.", now))


def test_the_model_is_told_little_and_its_reply_is_bounded():
    facts = morning_facts()
    system = n.system_prompt("morning", "TONE")
    assert "Use only what is in FACTS and the DRAFT" in system and "At most 6 short sentences" in system and system.endswith("TONE")
    user = n.user_prompt(facts, "DRAFT TEXT")
    assert user.startswith('FACTS: {"after":[{"at":"14:00"') and user.endswith("\nDRAFT: DRAFT TEXT") and len(user) < 1200
    assert json.loads(user.split("\n")[0][len("FACTS: "):]) == facts
    schema = n.reply_schema("now")
    assert schema["properties"]["spoken"]["maxLength"] == 340 and schema["required"] == ["spoken"]
    assert n.reply_text('{"spoken": " Hello there. "}') == "Hello there." and n.reply_text("<think>hm</think> Plain words.") == "Plain words."
    assert n.reply_text('{"spoken": 3}') == "" and n.reply_text("{broken") == ""


# --- the narrator, wired to the Day loop ---


class StubRuntime:
    """A model that rewrites the draft with `rewrite(draft)`, or raises. With `gate`, it answers only once the test
    sets the gate (a model that is slow for as long as the test says, not for a number of seconds)."""

    def __init__(self, rewrite=None, gate: threading.Event | None = None, error: Exception | None = None) -> None:
        self.rewrite, self.gate, self.error = rewrite or good, gate, error
        self.calls: list[dict] = []

    def chat(self, messages, tools=None, priority="chat", json_schema=None, on_text=None) -> ChatResult:
        self.calls.append({"messages": messages, "priority": priority, "json_schema": json_schema, "tools": tools, "on_text": on_text})
        if self.gate is not None:
            assert self.gate.wait(30), "the test never opened the gate"
        if self.error:
            raise self.error
        draft = messages[-1].content.split("\nDRAFT: ", 1)[1]
        return ChatResult(content=json.dumps({"spoken": self.rewrite(draft)}), model_version="stub-1")


def good(draft: str) -> str:
    """A faithful rewrite: the same facts, a warmer opening."""
    return "Right, here's where things stand. " + draft if not draft.startswith(("Morning", "Afternoon", "Evening")) else draft.replace(".", ", friend.", 1)


def invented(draft: str) -> str:
    return draft.replace("?", ", and your quiz is at half past four?") if "?" in draft else draft + " Your quiz is at half past four."


def build(items, places, tmp_path, runtime=None, narration=None, habits=None, with_hub=True) -> tuple[DayLoop, EventBus, list]:  # noqa: F811
    memory = FakeMemory(items=items)
    h = hub() if with_hub else ConnectorHub([], factory=never)
    habits = habits or FakeHabits(sleep=5.4)
    dayrun = DayRun(places, memory=memory, habits=habits, hub=h, data_dir=tmp_path, clock=lambda: at(9, 30))
    loop = DayLoop(dayrun, places, habits=habits, memory=memory, hub=h, clock=CLOCK, runtime=runtime, narration=narration,
                   description="A student with classes and a desk the box watches.")
    bus = EventBus(tmp_path / "events")
    loop.subscribe(bus)
    seen: list = []
    bus.subscribe("day.narrated", seen.append)
    return loop, bus, seen


async def settle() -> None:
    me = asyncio.current_task()
    while pending := [t for t in asyncio.all_tasks() if t is not me and not t.done()]:
        await asyncio.gather(*pending, return_exceptions=True)


def run(coro):
    return asyncio.run(coro)


async def plan_morning(loop: DayLoop, bus: EventBus, when: datetime | None = None) -> DayPlanProposed:
    events = loop.planner.plan(when or at(8, 10))
    await bus.publish_all(events)
    await settle()
    return events[0]


def by_moment(seen: list, moment: str) -> list[DayNarrated]:
    return [e for e in seen if e.moment == moment]


def test_the_fake_runtime_narrates_with_the_templates_and_never_calls_a_model(items, places, tmp_path):  # noqa: F811
    fake = FakeRuntime()
    loop, bus, seen = build(items, places, tmp_path, runtime=fake)
    assert loop.narrator.mode == "template" and loop.narrator.engine == "template"
    plan = run(plan_morning(loop, bus))
    (morning,) = by_moment(seen, "morning")
    assert morning.engine == "template" and morning.plan_id == plan.plan_id and morning.ref == plan.event_id and morning.day == plan.day
    assert morning.spoken.startswith("Morning. You got about five and a half hours, a little short:")
    assert morning.spoken.endswith("Want me to put this on your calendar?") and "Project Milestone 1 is due Tuesday" in morning.spoken
    assert "mentor check-in at twenty past nine" in morning.spoken and "Here's your day" not in morning.spoken
    assert "You slept 5.4 h" in morning.display and "9:20 AM" in morning.display
    assert by_moment(seen, "now") and by_moment(seen, "now")[0].engine == "template"  # the first status, in words
    assert not [c for c in fake.calls if c[0] == "chat"]
    # asked for a model without one: still the templates
    forced, _, _ = build(items, places, tmp_path / "b", runtime=fake, narration={"narration": "model"})
    assert forced.narrator.mode == "model" and forced.narrator.engine == "template"


def test_a_grounded_rewrite_from_the_model_is_what_gets_said(items, places, tmp_path):  # noqa: F811
    stub = StubRuntime()
    loop, bus, seen = build(items, places, tmp_path, runtime=stub)
    assert loop.narrator.mode == "model" and loop.narrator.engine == "model"  # the default with a real runtime
    run(plan_morning(loop, bus))
    (morning,) = by_moment(seen, "morning")
    assert morning.engine == "model" and morning.spoken.startswith("Morning, friend. You got about five and a half hours")
    assert "9:20 AM" in morning.display  # the screen keeps the template's exact paragraph
    (call,) = stub.calls  # one short, low-priority, schema-bounded call; the status that came with the plan used the template
    assert call["priority"] == "backfill" and call["tools"] is None and call["on_text"] is None
    assert call["json_schema"]["properties"]["spoken"]["maxLength"] == n.BOUNDS["morning"][1]
    system, user = call["messages"]
    assert system.role == "system" and "This person's words: \"study\"" in system.content and "A student with classes" in system.content
    assert user.content.startswith("FACTS: {") and "\nDRAFT: Morning. You got about five and a half hours" in user.content
    assert loop.narrator.stats["calls"] == 1 and loop.narrator.stats["used"] == 1
    assert by_moment(seen, "now")[0].engine == "template"


def test_an_invented_fact_a_timeout_and_a_runtime_error_all_fall_back_to_the_template(items, places, tmp_path):  # noqa: F811
    loop, bus, seen = build(items, places, tmp_path / "a", runtime=StubRuntime(invented))
    run(plan_morning(loop, bus))
    (morning,) = by_moment(seen, "morning")
    assert morning.engine == "template" and "quiz" not in morning.spoken and morning.spoken.startswith("Morning. You got")
    assert loop.narrator.stats["rejected"] == 1 and "half past four" in loop.narrator.stats["last_problem"]

    slow = StubRuntime(gate=threading.Event())
    loop, bus, seen = build(items, places, tmp_path / "b", runtime=slow, narration={"narration_timeout_s": 0.05})
    assert loop.narrator.config.timeout_s == 0.05

    async def until(ready, what: str) -> None:
        deadline = clocktime.monotonic() + 30.0  # only turns a bug into a failure instead of a hang
        while not ready():
            assert clocktime.monotonic() < deadline, what
            await asyncio.sleep(0.01)

    async def timed() -> None:
        await bus.publish_all(loop.planner.plan(at(8, 10)))
        await until(lambda: by_moment(seen, "morning"), "no morning narration")  # the template, while the model is still busy
        assert loop.narrator.inflight  # the runtime is still busy: the one slot stays taken
        await settle()
        slow.gate.set()  # the late reply lands now
        await until(lambda: not loop.narrator.inflight, "the late reply never freed the slot")

    run(timed())
    (morning,) = by_moment(seen, "morning")
    assert morning.engine == "template" and loop.narrator.stats["timeouts"] == 1
    assert not loop.narrator.inflight and loop.narrator.cache  # the late, grounded reply is kept for the next time these facts come up

    loop, bus, seen = build(items, places, tmp_path / "c", runtime=StubRuntime(error=RuntimeError("llama-server is down")))
    run(plan_morning(loop, bus))
    (morning,) = by_moment(seen, "morning")
    assert morning.engine == "template" and loop.narrator.stats["errors"] == 1 and not loop.narrator.inflight


def test_a_waiting_question_goes_first(items, places, tmp_path):  # noqa: F811
    stub = StubRuntime()
    loop, bus, seen = build(items, places, tmp_path, runtime=stub)

    async def scenario() -> None:
        await bus.publish(QuestionAsked(ts=at(8, 9), question_id="q1", text="where are my keys?"))
        await plan_morning(loop, bus)  # a question is waiting: the template, at once
        assert by_moment(seen, "morning")[-1].engine == "template" and stub.calls == []
        await bus.publish(AnswerGenerated(ts=at(8, 11), question_id="q1", answer="On the desk.", model_version="stub-1"))
        await bus.publish_all(loop.planner.plan(at(8, 20), remake=True))  # answered: the model may speak again
        await settle()

    run(scenario())
    assert by_moment(seen, "morning")[-1].engine == "model" and len(stub.calls) == 1
    # a question nobody answered stops holding the model back after two minutes of event time
    loop.narrator.questions["q2"] = at(8, 0)
    assert loop.narrator.question_waiting(at(8, 1)) and not loop.narrator.question_waiting(at(8, 3))


def test_a_reply_that_is_the_draft_word_for_word_is_counted_as_unchanged(items, places, tmp_path):  # noqa: F811
    stub = StubRuntime(lambda draft: draft)
    loop, bus, seen = build(items, places, tmp_path, runtime=stub)
    run(plan_morning(loop, bus))
    (morning,) = by_moment(seen, "morning")
    stats = loop.narrator.stats
    assert morning.engine == "template" and len(stub.calls) == 1
    assert (stats["calls"], stats["unchanged"], stats["used"], stats["rejected"], stats["timeouts"], stats["errors"]) == (1, 1, 0, 0, 0, 0)


def test_a_very_short_draft_is_never_sent_to_the_model(items, places, tmp_path):  # noqa: F811
    stub = StubRuntime()
    loop, bus, seen = build(items, places, tmp_path, runtime=stub)
    run(plan_morning(loop, bus))
    assert len(stub.calls) == 1  # the long morning brief was worth a rewrite
    later = at(8, 12)  # past the rate limit, so only the length decides
    job = loop.narrator.job_now("now", later)
    assert job is not None and len(job.draft.spoken) < MIN_DRAFT_CHARS == 120 and not loop.narrator.may_ask_model(job)
    said = run(loop.narrator.regenerate("now", later))
    assert said.engine == "template" and said.spoken == job.draft.spoken and len(stub.calls) == 1 and loop.narrator.stats["calls"] == 1
    # the brief seen on the box: two short sentences, nothing to improve
    short = n.Narration("Evening. Here is the plan: a long rest first, then dinner.", "A long rest first, then dinner.")
    morning = loop.narrator.job_now("morning", later)
    morning.hash, morning.draft = "other-facts", short
    assert not loop.narrator.may_ask_model(morning)
    morning.draft = n.Narration(short.spoken + " " + "And a longer stretch of study after that, until the team dinner at seven." * 2, short.display)
    assert loop.narrator.may_ask_model(morning)


def test_the_same_facts_are_generated_once_and_the_model_is_rate_limited(items, places, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.setattr(narrator_module, "MIN_DRAFT_CHARS", 0)  # this is about the rate limit; the status line here is a short one
    stub = StubRuntime()
    loop, bus, seen = build(items, places, tmp_path, runtime=stub, narration={"narration_gap_s": 60})
    client = TestClient(api(loop, bus, now=lambda: at(8, 10) + timedelta(seconds=30)))

    async def scenario() -> None:
        await plan_morning(loop, bus)
        assert len(stub.calls) == 1 and by_moment(seen, "morning")[-1].engine == "model"
        # the same plan again inside the minute: its facts differ (a new card), the model is not asked twice in 60 s of event time
        await bus.publish_all(loop.planner.plan(at(8, 10) + timedelta(seconds=40), remake=True))
        await settle()
        assert len(stub.calls) == 1

    run(scenario())
    # the same moment asked for again: the narration that exists, no event, no model
    again = client.post("/api/day/brief", json={"moment": "morning"}).json()
    assert again["fresh"] is False and len(stub.calls) == 1
    # more than a minute of event time later, a different moment may use the model again
    job = loop.narrator.job_now("now", at(8, 12))
    assert job is not None and loop.narrator.may_ask_model(job) and not loop.narrator.may_ask_model(loop.narrator.job_now("now", at(8, 10) + timedelta(seconds=50)))
    # an urgent status and an approval ask never wait for a model
    urgent = loop.narrator.job_now("now", at(8, 12))
    urgent.urgent = True
    assert not loop.narrator.may_ask_model(urgent)
    # facts the model already failed on are not asked about again
    loop.narrator.failed.add(job.hash)
    assert not loop.narrator.may_ask_model(job)


def test_a_restart_replays_the_narrations_without_calling_the_model(items, places, tmp_path):  # noqa: F811
    stub = StubRuntime()
    loop, bus, seen = build(items, places, tmp_path, runtime=stub)
    run(plan_morning(loop, bus))
    said = by_moment(seen, "morning")[-1]
    assert said.engine == "model" and len(stub.calls) == 1

    fresh = StubRuntime()
    again, bus2, seen2 = build(items, places, tmp_path, runtime=fresh)

    async def restart() -> None:
        assert await bus2.replay_into() > 0
        await settle()

    run(restart())
    assert fresh.calls == [] and seen2 == []  # nothing narrated on replay, nothing published
    assert again.narrator.latest["morning"] == said and again.narrator.latest["now"] == loop.narrator.latest["now"]
    assert again.narrator.cache[said.facts_hash] == said.spoken
    # the same facts after the restart use the words in the log, not the model
    job = again.narrator.job_for(again.planner.current[said.day])
    assert job.hash == said.facts_hash and again.narrator.say(job)[0].spoken == said.spoken and fresh.calls == []


def test_off_means_no_narration(items, places, tmp_path, monkeypatch):  # noqa: F811
    loop, bus, seen = build(items, places, tmp_path, runtime=StubRuntime(), narration={"narration": "off"})
    run(plan_morning(loop, bus))
    assert seen == [] and loop.narrator.engine == "off"
    assert TestClient(api(loop, bus)).post("/api/day/brief", json={"moment": "morning"}).status_code == 409
    monkeypatch.setenv("ATLAS_DAY_NARRATION", "template")  # the environment overrides agent.yaml
    assert NarrationConfig.of({"narration": "model"}, StubRuntime()).mode == "template"
    monkeypatch.delenv("ATLAS_DAY_NARRATION")
    assert NarrationConfig.of(None, StubRuntime()).mode == "model" and NarrationConfig.of(None, FakeRuntime()).mode == "template"
    assert NarrationConfig.of({"narration": "loud"}, None).mode == "template"
    cfg = NarrationConfig.of({"narration_timeout_s": 3, "narration_gap_s": 30, "narration_tone": "Dry."}, None)
    assert (cfg.timeout_s, cfg.gap_s, cfg.tone) == (3.0, 30.0, "Dry.")


def test_an_approval_card_is_asked_in_plain_words(items, places, tmp_path):  # noqa: F811
    loop, bus, seen = build(items, places, tmp_path, runtime=StubRuntime())

    async def scenario() -> None:
        plan, card = loop.planner.plan(at(8, 10))
        await bus.publish_all([plan, card])
        await bus.publish(ApprovalRequested(ts=at(8, 10), action_id=card.action_id, kind=card.kind, target=card.target, reason=card.reason,
                                            expires_at=at(8, 15)))
        await bus.publish(ApprovalRequested(ts=at(8, 11), action_id="dare", kind="circle_share", target="team",
                                            reason="Start a League dare: No energy drinks for 3 days, inviting Maya", expires_at=at(8, 16)))
        await bus.publish(ApprovalRequested(ts=at(8, 12), action_id="msg", kind="send_message", target="dom", reason="Package at the door",
                                            expires_at=at(8, 17)))
        await settle()
        return plan, card

    plan, card = run(scenario())
    asks = {e.ref: e for e in by_moment(seen, "approval")}
    count = sum(1 for m in plan.moves if not m.fixed)
    assert asks[card.action_id].spoken == f"I'd like to add {n.say_count(count, 'block')} to your calendar for today. Okay?"
    assert asks[card.action_id].engine == "template" and asks[card.action_id].plan_id == plan.plan_id  # an ask never waits for a model
    assert asks["dare"].spoken == "I'd like to start a League dare: no energy drinks for three days with Maya. Okay?"
    assert "msg" not in asks  # another context's request keeps its own reason
    assert set(loop.narrator.asks) == {card.action_id, "dare"}


def test_a_status_is_narrated_when_its_words_change_and_urgent_ones_are_marked(items, places, tmp_path):  # noqa: F811
    loop, bus, seen = build(items, places, tmp_path, runtime=FakeRuntime())

    async def scenario() -> DayStatusChanged:
        plan = await plan_morning(loop, bus)
        status = loop.tracker.status
        mentor = next(m for m in plan.moves if m.fixed and "mentor" in m.title)
        before = len(by_moment(seen, "now"))
        # only the meters moved: the same words, so nothing new is said
        await bus.publish(status.model_copy(update={"event_id": "s2", "ts": at(8, 12), "now": at(8, 12), "reason": "meters moved",
                                                    "meters": status.meters.model_copy(update={"meals": 2})}))
        assert len(by_moment(seen, "now")) == before
        nxt = status.next_event or loop.tracker.compute(at(8, 12)).next_event
        leave = nxt.model_copy(update={"id": mentor.id, "title": mentor.title, "start": mentor.start, "leave_by": mentor.start - timedelta(minutes=9),
                                       "walk_minutes": 9, "place_name": mentor.place_name})
        await bus.publish(status.model_copy(update={"event_id": "s3", "ts": mentor.start - timedelta(minutes=10), "now": mentor.start - timedelta(minutes=10),
                                                    "risk": "leave_now", "next_event": leave, "block_id": None, "block_title": None, "block_end": None,
                                                    "next_block": None, "drift_minutes": 0}))
        await settle()
        return status

    run(scenario())
    last = by_moment(seen, "now")[-1]
    assert last.urgent and last.engine == "template" and last.ref == "s3"
    assert last.spoken.startswith("Time to go. The mentor check-in starts at twenty past nine at Engineering II, a nine-minute walk from here; leave")
    assert not by_moment(seen, "now")[0].urgent


# --- the API and the push ---


def api(loop: DayLoop, bus: EventBus, now=lambda: at(8, 30)) -> FastAPI:
    app = FastAPI()
    app.include_router(day_router(loop, publish=bus.publish, clock=now))
    return app


def test_the_brief_api(items, places, tmp_path):  # noqa: F811
    loop, bus, seen = build(items, places, tmp_path, runtime=FakeRuntime())
    client = TestClient(api(loop, bus))
    empty = client.get("/api/day/brief").json()
    assert empty["mode"] == "template" and empty["engine"] == "template" and empty["latest"] == {} and empty["asks"] == {}
    assert client.post("/api/day/brief", json={"moment": "morning"}).status_code == 404  # no plan yet
    plan = run(plan_morning(loop, bus))
    body = client.get("/api/day/brief").json()
    assert set(body["latest"]) == {"morning", "now"} and body["timeout_s"] == 6.0 and body["gap_s"] == 60.0
    morning = body["latest"]["morning"]
    assert morning["type"] == "day.narrated" and morning["plan_id"] == plan.plan_id and morning["engine"] == "template" and morning["facts_hash"]
    one = client.get("/api/day/brief", params={"moment": "now"}).json()
    assert one["moment"] == "now" and one["narration"]["spoken"] == body["latest"]["now"]["spoken"]
    assert client.get("/api/day/brief", params={"moment": "coach"}).json()["narration"] is None
    assert client.get("/api/day/brief", params={"moment": "lunch"}).status_code == 422
    # "Brief me" twenty minutes on: the rest of the day from now, said again, published once
    before = len(seen)
    again = client.post("/api/day/brief", json={"moment": "morning", "fast": True}).json()
    assert again["moment"] == "morning" and again["narration"]["spoken"].startswith("Morning.")
    if again["fresh"]:
        assert len(seen) == before + 1 and loop.narrator.latest["morning"].event_id == again["narration"]["event_id"]
    repeat = client.post("/api/day/brief", json={"moment": "morning", "fast": True}).json()
    assert repeat["fresh"] is False and repeat["narration"]["event_id"] == again["narration"]["event_id"] and len(seen) <= before + 1
    assert client.post("/api/day/brief", json={"moment": "now"}).json()["narration"]["moment"] == "now"
    assert client.post("/api/day/brief", json={"moment": "evening"}).status_code == 404  # no review yet
    assert client.post("/api/day/brief", json={"moment": "approval"}).status_code == 422


class FakeNtfy:
    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []

    async def publish(self, topic, message, title=None, tags=None) -> None:
        self.sent.append((title, message))


def test_the_push_uses_the_narrators_words_once(items, places, tmp_path):  # noqa: F811
    loop, bus, seen = build(items, places, tmp_path, runtime=FakeRuntime())
    ntfy = FakeNtfy()
    bus.subscribe("day.", day_phone(ntfy, "atlas", narrated=True))
    plan = run(plan_morning(loop, bus))
    assert ntfy.sent == [("Your day", loop.narrator.latest["morning"].display)]  # not the raw plan text as well, and not every status
    urgent = loop.narrator.latest["now"].model_copy(update={"event_id": "u1", "urgent": True, "display": "Leave now: judging starts at 3:30 PM."})
    run(bus.publish(urgent))
    assert ntfy.sent[-1] == ("Time to move", "Leave now: judging starts at 3:30 PM.")
    raw = FakeNtfy()
    push = day_phone(raw, "atlas")  # narration off: the events' own texts, as before
    run(push(plan))
    run(push(loop.narrator.latest["morning"]))
    assert raw.sent == [("Your day", plan.text)]


def test_the_day_in_words_for_a_conversation_is_todays_and_reads_only(items, places, tmp_path):  # noqa: F811
    loop, bus, seen = build(items, places, tmp_path, runtime=FakeRuntime())
    assert loop.in_words(at(8, 0)) == {}  # no plan yet: nothing to say about the day
    run(plan_morning(loop, bus))
    before = len(seen)
    words = loop.in_words(at(10, 0))
    assert words["plan"] == loop.narrator.latest["morning"].display
    assert words["now"] and words["now"] == loop.narrator.job_now("now", at(10, 0)).draft.display
    assert "coach" not in words and "review" not in words and len(seen) == before  # nothing published
    assert loop.in_words(at(10, 0) + timedelta(days=2)) == {}  # another day: the old plan is not today's


def test_the_leave_by_never_lands_inside_a_place_name_with_an_initial():
    from contexts.day.domain.narration import _weave

    shown = _weave("A nap before the mentor check-in at 3:14 PM (John C. Hitt Library). Later: demos 5:54 PM.", " 13 min walk: leave by 3:02 PM.")
    assert shown == "A nap before the mentor check-in at 3:14 PM (John C. Hitt Library). 13 min walk: leave by 3:02 PM. Later: demos 5:54 PM."
    assert _weave("One thing only", " 5 min walk: leave by 1:00 PM.") == "One thing only. 5 min walk: leave by 1:00 PM."
