# OWNER'S RULE, part two (Hemdutt). Three tiers: small talk is one or two short lines; a FUNCTIONAL question ("what does my
# day look like", "what's next", "what did you do for me today") is a real task answered from live data and said like a
# person; a complex request takes the full planner and tool loop, unchanged.
# "We can't have hard-coded answers, because Atlas has to be prepared for anything": the facts are always fetched live, the
# sentences built in code are only a FAST LANE for the phrasings the patterns recognise, and any other question about the
# day goes to the model with the same facts. These hold what the Jetson got wrong after #368; if one fails, fix the path.
import asyncio
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import yaml

from contexts.agent.application import schedule
from contexts.agent.tests.test_natural_speech import DAY, NOW, RECORD, RUNDOWN, Recorder, ask, box, item, sentences
from contracts.memory import Item

# config/agent.yaml `listings:` as the gateway loads it: nothing about UCF is written in code
LISTINGS = schedule.Listings.of(yaml.safe_load(Path("config/agent.yaml").read_text())["listings"])
NIGHT = datetime(2026, 10, 11, 5, 37, tzinfo=UTC)  # Sunday 1:37 AM, when the box was asked


def campus(i, title, hour, minute, until, place, slug):
    """One event exactly as the box stores it. The campus feed is one of the ics connector's feeds
    (config/connectors.yaml `calendar.options.feeds: {name: campus, url: https://events.ucf.edu/upcoming/feed.ics}`), so
    contexts/connectors/servers/ics.py `expand` and the hub make it: connector "calendar", kind "calendar_event" (the
    same as the person's own classes), external id "campus:<uid>:<start>", the event's own page as the url, and
    "until <end> · <location>" as the summary. Nothing in it says "optional"."""
    start = datetime(2026, 10, 11, 4, 0, tzinfo=UTC) + timedelta(hours=hour, minutes=minute)
    return Item(connector="calendar", kind="calendar_event", external_id=f"campus:{3900000 + i}@events.ucf.edu:{start.isoformat()}",
                title=title, when=start, url=f"https://events.ucf.edu/event/{3900000 + i}/{slug}/", summary=f"until {until} · {place}")


# the five the box read out as the person's own on Sunday
CAMPUS = [campus(0, "U Can Finish Race", 7, 30, "10:30 AM", "Memory Mall", "u-can-finish-race"),
          campus(1, "Theatre UCF Open House", 11, 0, "4:00 PM", "Performing Arts Center", "theatre-ucf-open-house"),
          campus(2, "Fall Climbing Competition: Dungeons & Dynos", 12, 0, "5:00 PM", "RWC Climbing Tower", "fall-climbing-competition-dungeons-dynos"),
          campus(3, "UCF Women's Soccer vs. Oklahoma State", 13, 0, "3:00 PM", "UCF Soccer and Track Stadium", "ucf-womens-soccer-vs-oklahoma-state"),
          campus(4, "Graduate School Mixer", 15, 0, "5:00 PM", "Student Union", "graduate-school-mixer")]
HW = item("a2", "HW 4: linked lists [COP3502]", 23, 59, kind="assignment", connector="canvas", url="https://canvas/hw4")
# DayLoop.blocks, as the gateway wires it into `day_plan`: the plan's chosen blocks in the narrator's words, and when to leave
PLAN = [{"kind": "study", "title": "a study block", "start": NOW + timedelta(minutes=15), "end": NOW + timedelta(minutes=80)},
        {"kind": "meal", "title": "lunch", "start": NOW + timedelta(hours=3), "end": NOW + timedelta(hours=3, minutes=30)},
        {"kind": "leave", "title": "COP4533 lecture", "start": NOW + timedelta(minutes=90)}]
# Six things of their own from 9:00 (two plan blocks, four events), one thing due, five listings. A walk-through says ALL of it.
WHOLE_DAY = ("Big day. This morning you've got a study block at 9:15, then your COP 4533 lecture at 10:40. Leave by 10:30 for that one. "
             "This afternoon it's lunch at noon, your PHY 2049 lecture at 2, then the mentor check-in at 4:30 PM. Tonight there's Dinner with Maya at 7. "
             "HW 4: linked lists is due today at 11:59 PM. If you want a break, there are five things on campus, like Theatre UCF Open House at 11 "
             "and Dungeons & Dynos at noon. Does that look right, or should I change something?")
# A casual ask about the same day is short: the first two things, the first thing due, the listings counted, an offer of the rest.
SHORT_DAY = ("You've got six things today, starting with a study block at 9:15, then your COP 4533 lecture at 10:40. "
             "Leave by 10:30 for that one. HW 4: linked lists is due today at 11:59 PM, and there are five things on campus if you want them. "
             "Want the rest?")
OWN = ("a study block at 9:15", "your COP 4533 lecture at 10:40", "lunch at noon", "your PHY 2049 lecture at 2", "the mentor check-in at 4:30 PM",
       "Dinner with Maya at 7")
# The two groups (schedule.py `Ask.whole`). "What's next?" is neither: it is one thing.
WALK_THROUGH = ["What does my day look like?", "run me through my day", "walk me through my day", "brief me", "what's my day", "give me the rundown",
                "What do I have today?", "what have I got today", "Hey Atlas, what does my day look like today?", "how's my day looking?",
                "what's the plan for today?", "give me a rundown of my day"]
CASUAL_DAY = ["what's on today?", "anything today?", "what is on today", "anything on today?"]
CASUAL_CALENDAR = ["what's on my calendar?", "what's on my calendar today?", "do I have anything this afternoon?", "what's coming up?"]
DONE_FULL = "Today I've planned your day and reminded you twice. I've also answered six questions. What else can I do for you?"


def the_box(items, at=NOW, plan=None, runtime=None, listings=LISTINGS, done=None):
    qa, rt = box(runtime, items=items) if done is None else box(runtime, items=items, done=done)
    qa.clock = qa.talk.clock = lambda: at
    qa.listings = qa.talk.listings = listings
    qa.talk.day_facts = None  # the narrator's own sentences are not what is under test here
    if plan is not None:
        qa.day_plan = qa.talk.day_plan = lambda: plan
    return qa, rt


# --- (1) campus listings are never the person's own commitments ---

def test_the_real_boxs_events_ucf_edu_items_are_recognised_from_their_own_data():
    assert all(LISTINGS.listed(i) for i in CAMPUS)
    assert LISTINGS.hosts == ("events.ucf.edu",) and LISTINGS.feeds == ("campus",)  # from config, not from code
    assert schedule.Listings().hosts == () and schedule.Listings().feeds == ()
    # each signal on its own is enough: the feed's name in the external id, or where the link points (subdomains too)
    by_feed = Item(connector="calendar", kind="calendar_event", external_id="campus:1:2026", title="Movie night", when=NOW)
    by_host = Item(connector="calendar", kind="calendar_event", external_id="x:1", title="Movie night", when=NOW, url="https://www.events.ucf.edu/event/1/movie/")
    assert LISTINGS.listed(by_feed) and LISTINGS.listed(by_host)
    # the person's own calendar is theirs, whatever it links to; so is a listed event they said they are going to
    own = Item(connector="calendar", kind="calendar_event", external_id="outlook:abc:2026", title="COP4533 lecture", when=NOW,
               url="https://outlook.office365.com/owa/?itemid=1", summary="until 11:30 AM · L3Harris Engineering Center")
    going = CAMPUS[1].model_copy(update={"summary": "until 4:00 PM · Performing Arts Center · attending"})
    assert not LISTINGS.listed(own) and not LISTINGS.listed(going)
    assert not LISTINGS.listed(Item(connector="calendar", kind="calendar_event", external_id="outlook:1", title="Lunch", when=NOW,
                                    url="https://notevents.ucf.edu.example.com/x"))


def test_without_the_listing_sources_the_fixture_reproduces_what_the_box_said():
    """After #368 (which only knew the `site`/`search`/`groupme` connectors and the word "optional"), on the real box."""
    qa, _ = the_box(CAMPUS, NIGHT, listings=schedule.Listings())
    assert asyncio.run(qa.ask("What's on my calendar today?")).text == (
        "You've got five things today. First is U Can Finish Race at 7:30, then Theatre UCF Open House at 11. Want the rest?")
    assert asyncio.run(qa.ask("Am I free at 3?")).text == "No, you've got Theatre UCF Open House then, until 4."
    assert asyncio.run(qa.ask("What's next?")).text == "Next up is U Can Finish Race at 7:30."


def test_campus_listings_are_on_campus_never_yours_and_never_make_you_busy():
    qa, rt = the_box(CAMPUS, NIGHT, runtime=Recorder("You have five things today."))
    a = asyncio.run(qa.ask("What's on my calendar today?"))
    assert a.text == "There are five things on campus today. First is U Can Finish Race at 7:30, then Theatre UCF Open House at 11. Want the rest?"
    assert [c.url for c in a.citations] == [i.url for i in CAMPUS]  # still cited by their links
    assert asyncio.run(qa.ask("Am I free at 3?")).text == "Yes, you're free at 3 this afternoon. There's Theatre UCF Open House on campus then if you want it."
    assert asyncio.run(qa.ask("am I busy at 3?")).text == "No, you're free at 3 this afternoon. There's Theatre UCF Open House on campus then if you want it."
    assert asyncio.run(qa.ask("What's next?")).text == "Nothing of yours is next. On campus there's U Can Finish Race at 7:30."
    assert asyncio.run(qa.ask("What does my day look like?")).text == (
        "Nothing of yours is on today. There are five things on campus, like U Can Finish Race at 7:30 and Theatre UCF Open House at 11. "
        "Want me to plan your day?")
    assert rt.seen == []  # all from the data
    for question in ("What's on my calendar today?", "Am I free at 3?", "What's next?", "What does my day look like?", "do I have anything today?"):
        assert not re.search(r"\byou(?:'ve| have) got\b|\bNext up\b|^No\b", asyncio.run(qa.ask(question)).text), question


def test_whats_next_prefers_the_persons_own_next_thing_over_a_listing():
    qa, _ = the_box(CAMPUS + DAY, NIGHT)  # the race is at 7:30, their lecture at 10:40
    assert asyncio.run(qa.ask("What's next?")).text == "Next up is your COP 4533 lecture at 10:40."
    assert asyncio.run(qa.ask("Am I free at 3?")).text == "No, you've got your PHY 2049 lecture then, until 3:15 PM."  # their own class still does
    qa, _ = the_box(CAMPUS + DAY, plan=PLAN)
    assert ask(qa, "What's next?") == "Next up is a study block at 9:15."  # a block of the agreed plan is theirs
    qa, _ = the_box(CAMPUS + [HW], NOW + timedelta(hours=11))  # 8 PM: every listing is over, the homework is not
    assert ask(qa, "What's next?") == "Next up is HW 4: linked lists, due today at 11:59 PM."  # a deadline is theirs
    qa, _ = the_box(CAMPUS, NOW + timedelta(hours=11))
    assert ask(qa, "What's next?") == "Nothing else on your calendar today."


# --- (2) "today" is not always the cameras' ---

DONE_ASKS = ["What did you do for me today?", "what have you done for me", "what have you been doing for me", "anything you did for me today",
             "Hey Atlas, what did you do for me today?", "what have you done for me today?", "have you done anything for me today"]
RUNDOWN_ASKS = ["What do I have today?", "what have I got today", "what's my day", "What does my day look like?", "run me through my day",
                "what do i have today", "what's my day today"]
HELLO_ASKS = ["How are you doing today?", "how are you today", "how's it going today?", "Hey Atlas, how are you today?"]
CAMERA_ASKS = ["what did you see today", "What happened today?", "what did I do today", "what did you see today on the desk camera?",
               "did anyone come by today?", "what happened while I was out today?"]


@pytest.mark.parametrize("question", DONE_ASKS)
def test_what_did_you_do_for_me_is_its_work_for_you_not_the_camera_log(question):
    for runtime in (None, Recorder("Sat 8:00 PM: I saw a busy indoor atrium. Sat 9:00 PM: ...")):
        qa, rt = box(runtime)
        said = ask(qa, question)
        assert said == DONE_FULL and getattr(rt, "seen", []) == []  # from real events (done.py), with or without a model
        assert qa.prefetch(question) is None and qa.talk.owns(question)  # the camera recap never takes it
        assert 2 <= len(sentences(said)) <= 3 and said.endswith("?") and not RECORD.search(said)
        assert "rehearse" not in said and "saw" not in said
    quiet, _ = box(done=[])
    assert ask(quiet, question) == "Nothing yet today, it's been quiet. What can I do for you?"
    busy, _ = box(done=["re-planned your day", "left one thing waiting for your OK", "sent the message you approved", "reminded you twice",
                        "checked in on you", "answered six questions"])
    assert ask(busy, question) == ("Today I've re-planned your day and left one thing waiting for your OK. I've also sent the message you approved, "
                                   "reminded you twice and checked in on you. What else can I do for you?")


@pytest.mark.parametrize("question", RUNDOWN_ASKS)
def test_what_do_i_have_today_is_the_days_rundown_not_the_camera_recap(question):
    for runtime in (None, Recorder("Sun 8:00 AM hour: two people rehearse a presentation.")):
        qa, rt = box(runtime)
        qa.talk.day_facts = None
        a = asyncio.run(qa.ask(question))
        assert a.text == RUNDOWN and getattr(rt, "seen", []) == []
        assert all(c.url is None and c.episode_id is None and c.frame_ref is None for c in a.citations)  # calendar items, not frames
    empty, _ = box(items=[])
    empty.talk.day_facts = None
    assert asyncio.run(empty.ask(question)).text == "Nothing of yours is on today. Want me to plan your day?"  # never the cameras


@pytest.mark.parametrize("question", HELLO_ASKS)
def test_how_are_you_doing_today_is_small_talk(question):
    qa, rt = box(Recorder("Sun 8:00 AM hour: two people rehearse a presentation."))
    assert ask(qa, question) == "Good. I've planned your day and reminded you twice so far. What else can I do for you?"
    assert rt.seen == [] and qa.prefetch(question) is None and qa.talk.owns(question)


@pytest.mark.parametrize("question", CAMERA_ASKS)
def test_a_genuine_camera_question_about_today_stays_with_the_cameras(question):
    qa, rt = box(Recorder("Two people rehearsed a presentation around 8:40."))
    assert not qa.talk.owns(question)
    ev = qa.prefetch(question)
    if "while i was out" in question.lower() or "anyone" in question.lower():  # these were the time window's or the tool loop's already
        assert ev is None or ev.citations[0].episode_id == "e1"
        return
    assert ev is not None and ev.direct is None and [c.episode_id for c in ev.citations] == ["e1"]  # what the cameras saw today
    a = asyncio.run(qa.ask(question))
    assert a.text == "Two people rehearsed a presentation around 8:40." and "Evidence:" in rt.seen[0][1].content
    assert "rehearse a presentation" in rt.seen[0][1].content and a.citations[0].episode_id == "e1"


# --- (3) a day rundown is the whole day: own events, the plan's blocks, what is due, then the listings ---

@pytest.mark.parametrize("question", WALK_THROUGH)
def test_a_walk_through_says_the_whole_day_by_part_of_the_day_with_no_cap(question):
    qa, rt = the_box(CAMPUS + DAY + [HW], plan=PLAN, runtime=Recorder("You have a lot on."))
    a = asyncio.run(qa.ask(question))
    said = a.text
    assert said == WHOLE_DAY and rt.seen == []
    assert all(thing in said for thing in OWN)  # every one of the six things that are theirs: nothing dropped, nothing offered "later"
    assert "Want the rest" not in said and said.endswith("should I change something?")  # an offer about the plan instead
    assert [said.index(t) for t in OWN] == sorted(said.index(t) for t in OWN)  # one order: the clock's
    assert said.index("This morning") < said.index("This afternoon") < said.index("Tonight") < said.index("is due") < said.index("on campus")
    assert not RECORD.search(said) and "U Can Finish Race" not in said  # no list read out; 7:30 is behind us at 9
    # the campus things stay capped (they are not theirs): how many, and two of them
    assert sum(1 for i in CAMPUS if i.title.split(": ")[-1] in said) == 2 and "five things on campus" in said
    assert qa.talk.rest is None  # nothing was held back, so a later "yes" is just a yes
    assert len(a.citations) == 10  # the items behind it are still cited


@pytest.mark.parametrize("question", CASUAL_DAY)
def test_a_casual_question_about_the_same_day_is_capped_with_an_offer_of_the_rest(question):
    qa, rt = the_box(CAMPUS + DAY + [HW], plan=PLAN, runtime=Recorder("You have a lot on."))
    said = ask(qa, question)
    assert said == SHORT_DAY and rt.seen == []
    assert sum(1 for thing in OWN if thing in said) == 2 and len(sentences(said)) <= 4  # two things, the first due, then the offer
    rest = ask(qa, "yes")  # and the rest is all of it
    assert rest == ("You've got a study block at 9:15, your COP 4533 lecture at 10:40, lunch at noon, your PHY 2049 lecture at 2, "
                    "the mentor check-in at 4:30 PM, then Dinner with Maya at 7 tonight. Leave by 10:30 for your COP 4533 lecture. "
                    "HW 4: linked lists is due today at 11:59 PM. On campus there's Theatre UCF Open House at 11, Dungeons & Dynos at noon, "
                    "UCF Women's Soccer vs. Oklahoma State at 1, Graduate School Mixer at 3, then Board game night at 8 tonight.")


@pytest.mark.parametrize("question", CASUAL_CALENDAR)
def test_a_narrow_calendar_question_stays_short(question):
    qa, _ = the_box(CAMPUS + DAY + [HW], plan=PLAN)
    said = ask(qa, question)
    assert len(sentences(said)) <= 3 and sum(1 for t in ("lecture at", "check-in", "Dinner", "study block", "lunch") if t in said) <= 3
    assert "Big day" not in said and "This morning" not in said
    qa, _ = the_box(DAY)
    assert ask(qa, "What's next?") == "Next up is your COP 4533 lecture at 10:40."  # one thing, always


def test_only_an_absurd_day_is_shortened_and_never_by_dropping_a_commitment():
    events = [item(f"m{h}", f"Meeting {chr(65 + h - 9)}", h, 30) for h in range(9, 19)]  # ten meetings of their own
    blocks = [{"kind": "study", "title": "a short study block", "start": NOW + timedelta(hours=h, minutes=5), "end": NOW + timedelta(hours=h, minutes=25)}
              for h in (1, 2, 4, 5)]
    qa, _ = the_box(events, plan=blocks)  # fourteen things: past the ceiling
    said = ask(qa, "run me through my day")
    assert all(f"Meeting {chr(65 + i)}" in said for i in range(10))  # every commitment is still named
    assert "study block" not in said and said.count("plan blocks in between") == 2  # the small ones are counted, not listed
    assert "with two plan blocks in between" in said and "Want the rest" not in said
    qa, _ = the_box(events[:8], plan=blocks)  # twelve: all of it, blocks too
    assert ask(qa, "run me through my day").count("a short study block") == 4


def test_the_walk_through_is_read_live_every_time_never_a_kept_string():
    plan = list(PLAN)
    qa, _ = the_box([DAY[0]], plan=plan)
    assert ask(qa, "what does my day look like?") == ("This morning you've got a study block at 9:15, then your COP 4533 lecture at 10:40. "
                                                      "Leave by 10:30 for that one. This afternoon it's lunch at noon. Does that look right, or should I change something?")
    plan[:] = [{"kind": "study", "title": "a long study block", "start": NOW + timedelta(hours=2), "end": NOW + timedelta(hours=4)}]  # a re-plan
    assert ask(qa, "what does my day look like?") == ("This morning you've got your COP 4533 lecture at 10:40, then a long study block at 11. "
                                                      "Does that look right, or should I change something?")
    qa.day_plan = qa.talk.day_plan = lambda: None  # no plan any more, and a port that fails must not break the answer
    assert ask(qa, "what does my day look like?") == "This morning you've got your COP 4533 lecture at 10:40. Want me to plan your day?"
    qa.day_plan = qa.talk.day_plan = lambda: 1 / 0
    assert ask(qa, "what does my day look like?") == "This morning you've got your COP 4533 lecture at 10:40. Want me to plan your day?"


def test_yes_to_want_me_to_plan_your_day_starts_the_day_loops_own_plan_and_its_card():
    qa, _ = the_box([])
    assert ask(qa, "what does my day look like?") == "Nothing of yours is on today. Want me to plan your day?"
    assert qa.plans_started == []
    assert ask(qa, "yes") == "Okay, I've drafted a plan for today. Have a look on Today's Plan; nothing goes on your calendar until you approve it."
    assert qa.plans_started == [NOW]  # the gateway's port: DayLoop.planner.plan, which proposes the plan and its approval card
    assert ask(qa, "yes") != ask(qa, "what does my day look like?") and qa.plans_started == [NOW]  # a later "yes" is just a yes
    ask(qa, "what does my day look like?")
    ask(qa, "tell me a joke")
    ask(qa, "yes")
    assert qa.plans_started == [NOW]  # the offer does not outlive the next thing said
    ask(qa, "what does my day look like?")
    qa.clock = qa.talk.clock = lambda: NOW + timedelta(minutes=10)
    ask(qa, "yes")
    assert qa.plans_started == [NOW]  # nor a long silence

    async def plan_day():  # the real port is async (it publishes on the bus)
        qa.plans_started.append("async")

    qa, _ = the_box([])
    qa.plans_started, qa.talk.plan_day = [], plan_day
    ask(qa, "what do I have today?")
    assert ask(qa, "sure") .startswith("Okay, I've drafted a plan") and qa.plans_started == ["async"]
    # a box whose conversation cannot start a plan never offers one: the line is an offer that works
    qa, _ = the_box([])
    qa.talk.plan_day = None
    assert ask(qa, "what does my day look like?") == "Nothing of yours is on today. I can plan it with you on Today's Plan."
    qa.talk.plan_day = lambda: 1 / 0  # and a plan that fails to start says so
    ask(qa, "what does my day look like?")
    assert ask(qa, "yes") == "I couldn't start a plan just now. I can plan it with you on Today's Plan."


def test_with_no_plan_and_nothing_of_their_own_it_says_so_and_offers_to_plan():
    qa, _ = the_box([])
    assert ask(qa, "what does my day look like?") == "Nothing of yours is on today. Want me to plan your day?"
    qa, _ = the_box([HW])
    assert ask(qa, "what do I have today?") == "Nothing of yours is on today. HW 4: linked lists is due today at 11:59 PM. Want me to plan your day?"
    qa, _ = the_box(DAY, NOW + timedelta(hours=14), plan=[])  # 11 PM with a plan: the day is done, nothing to plan
    assert ask(qa, "what does my day look like?") == "Nothing else of yours today."
    # a leave-by time for something that is only listed is not theirs to leave for
    qa, _ = the_box(CAMPUS[1:], plan=[{"kind": "leave", "title": "Theatre UCF Open House", "start": NOW + timedelta(hours=1)}])
    assert "Leave by" not in ask(qa, "what does my day look like?")


UNRECOGNISED = ["is today going to be rough?", "when can I fit the gym in?", "what should I skip?", "will I have time for a nap before class?",
                "how packed is my afternoon, honestly?"]


@pytest.mark.parametrize("question", UNRECOGNISED)
def test_any_other_question_about_the_day_goes_to_the_model_with_the_same_live_facts(question):
    qa, rt = the_box(CAMPUS + DAY + [HW], plan=PLAN, runtime=Recorder("Honestly it's a full one, but the gym fits after physics."))
    assert schedule.parse(question, NOW, qa.sleep.tz) is None or "class" in question  # no pattern answers it
    assert ask(qa, question) == "Honestly it's a full one, but the gym fits after physics."  # the model's words, not a canned line
    facts = rt.seen[0][0].content.split("FACTS\n")[1].split("\n")
    assert rt.seen[0][-1].content == question and len(rt.seen) == 1
    # each kind is there, in the wording for the ear, and the listings are marked as optional
    assert "- Their own events for the rest of today: your COP 4533 lecture at 10:40, your PHY 2049 lecture at 2, the mentor check-in at 4:30 PM, Dinner with Maya at 7 tonight." in facts
    assert "- Blocks of the day plan they agreed to: a study block at 9:15, lunch at noon." in facts
    assert "- To be on time for your COP 4533 lecture they should leave by 10:30." in facts
    assert "- Due today or tomorrow: HW 4: linked lists today at 11:59 PM." in facts
    listed = next(f for f in facts if "campus" in f)
    assert listed == ("- Optional campus events today (public listings, NOT their commitments; they are free during these): Theatre UCF Open House at 11, "
                      "Dungeons & Dynos at noon, UCF Women's Soccer vs. Oklahoma State at 1, Graduate School Mixer at 3, Board game night at 8 tonight.")
    own = next(f for f in facts if f.startswith("- Their own events"))
    assert not any(i.title.split(":")[-1].strip() in own for i in CAMPUS)  # a listing is never among their own
    assert not RECORD.search("\n".join(facts).replace("(public listings, NOT their commitments; they are free during these)", ""))


def test_the_facts_say_so_when_a_kind_is_empty_and_small_talk_still_gets_none_of_it():
    qa, _ = the_box([])
    facts = qa.talk.facts("is today going to be rough?")
    assert "They have no events of their own for the rest of today." in facts and "Nothing is due today or tomorrow." in facts
    assert not any("campus" in f or "plan they agreed" in f for f in facts)
    qa, _ = the_box(CAMPUS + DAY + [HW], plan=PLAN)
    for chat in ("tell me a joke", "how's it going?", "what's the capital of France?", "what did you see today?", "what did you do for me today?"):
        assert not any("own events" in f or "campus" in f or "Due today" in f for f in qa.talk.facts(chat)), chat


def test_with_no_model_an_unrecognised_day_question_still_gets_the_live_rundown():
    qa, _ = the_box(CAMPUS + DAY + [HW], plan=PLAN)  # the fake runtime on a laptop
    assert ask(qa, "is today going to be rough?") == SHORT_DAY
