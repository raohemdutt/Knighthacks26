# OWNER'S RULE (Hemdutt): Atlas speaks like a person and says only what was asked. "How's it going?" is "Good. I've <its
# work for you>. What else can I do for you?"; the calendar is sentences, never "Coming up today: A, Sun 7:30 AM; B, ...".
# If one of these fails, the spoken path has changed back: fix the path, do not loosen the test.
import asyncio
import re
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from contexts.agent.application.done import DoneToday
from contexts.agent.application.memory_port import FakeMemory
from contexts.agent.application.qa import NO_EVIDENCE, QA
from contexts.agent.application.talk import Talk, TalkConfig
from contracts.memory import Episode, Item
from contracts.types import ChatResult, ModelVersion, ToolCall
from runtime import FakeRuntime

NOW = datetime(2026, 10, 11, 13, 0, tzinfo=UTC)  # Sunday 9:00 AM in New York
CAPTION = "Two people rehearse a presentation; one points at the monitor."
TITLES = ("COP4533", "COP 4533", "PHY2049", "PHY 2049", "lecture", "mentor check-in", "Dinner with Maya", "Board game night", "Milestone")
RECORD = re.compile(r"\b[AP]M to\b|\d ?(?:-|–|to) ?\d{1,2}(?::\d\d)? ?[AP]M|[\[\]()]|\bcam-?\w*\d*\b|\b(?:Sun|Mon|Tue|Wed|Thu|Fri|Sat) \d|;|Coming up"
                    r"|\bcalendar_event\b|\bfake\b|Evidence:", re.I)


def item(id, title, hour, minute=0, days=0, summary=None, kind="calendar_event", connector="calendar", url=None):
    when = datetime(2026, 10, 11, 4, 0, tzinfo=UTC) + timedelta(days=days, hours=hour, minutes=minute)  # New York is UTC-4
    return Item(connector=connector, kind=kind, external_id=id, title=title, when=when, summary=summary, url=url)


DAY = [
    item("c1", "COP4533 lecture", 10, 40, summary="until 11:30 AM · L3Harris Engineering Center"),
    item("c2", "PHY2049 lecture", 14, summary="until 3:15 PM · Physical Sciences 161"),
    item("c3", "Knight Hacks: mentor check-in", 16, 30, summary="until 5:00 PM · Engineering II atrium"),
    item("c4", "Dinner with Maya", 19),
    item("c5", "Board game night", 20, summary="until 10:00 PM · Student Union · with Riley · optional · via UCF Events"),
    item("a1", "Project - Milestone1 Fall2026 [COP4533]", 15, 55, days=2, kind="assignment", connector="canvas", url="https://canvas/m1",
         summary="Due Tue Oct 13, 3:55 PM. 100 points."),
]
SEEN = [Episode(id="e1", camera_id="cam-desk", start=NOW - timedelta(minutes=20), end=NOW - timedelta(minutes=18), labels=["person"],
                summary=CAPTION, keyframe_refs=["frames:cam-desk:1"])]
THEIR_DAY = {"now": "On plan: study block until 10:20 AM. The COP4533 lecture is at 10:40 AM at L3Harris (10 min walk, leave by 10:30 AM).",
             "now_spoken": "You're on plan: your study block runs until twenty past ten. The lecture is at twenty to eleven.",
             "plan": "First up is COP4533 lecture at 10:40 AM (L3Harris). Later: PHY2049 lecture 2 PM. Bed at 11 PM.",
             "coach": "You did 45 of 50 minutes. One break."}
DONE = ["planned your day", "reminded you twice", "answered six questions"]


class Recorder:
    """A real model's place: records every prompt it is sent and says what it was told to say."""

    def __init__(self, *replies):
        self.replies, self.seen = list(replies), []

    def chat(self, messages, tools=None, priority="chat", json_schema=None, on_text=None):
        self.seen.append(messages)
        r = self.replies.pop(0) if self.replies else "Sure."
        r = r if isinstance(r, ChatResult) else ChatResult(content=r, model_version="atlas-v9")
        if on_text and r.content:
            on_text(r.content)
        return r

    def model_info(self):
        return ModelVersion(version="atlas-v9", base="stub")


def box(runtime=None, items=DAY, names=None, done=DONE, web=None):
    """QA with the conversation wired the way the gateway wires it, on a seeded day."""
    rt = runtime or FakeRuntime()
    memory = FakeMemory(episodes=SEEN, items=list(items))
    qa = QA(rt, memory, clock=lambda: NOW, web=web)
    qa.talk = Talk(rt, qa.clock, memory=memory, config=TalkConfig(names=tuple((names or {}).items())), tz=qa.sleep.tz, web=bool(web))
    qa.names = dict(names or {})
    qa.talk.day_facts = lambda: dict(THEIR_DAY)
    qa.talk.done_for_you = lambda: list(done)
    qa.talk.about = "A student with classes, assignments on Canvas and a desk the box watches."
    return qa, rt


def ask(qa, question):
    pieces = []
    a = asyncio.run(qa.ask(question, stream=pieces.append))
    assert "".join(pieces).strip() == a.text  # what is streamed (and so spoken) is the answer, once
    return a.text


def sentences(text):
    return [s for s in re.split(r"(?<=[.!?])\s+", text) if s]


def nothing_volunteered(text):
    assert not any(t.lower() in text.lower() for t in TITLES), text  # none of today's events
    assert "rehearse" not in text and "presentation" not in text and "moment" not in text and "keyframe" not in text  # nothing the cameras saw
    assert not re.search(r"\bAI\b|\brunning for\b|\b\d{1,2}:\d\d\b|\b[AP]M\b|Sunday", text), text  # not what it is, its uptime or the time


# --- (1) hello: one short line, built in code, nothing volunteered ---

@pytest.mark.parametrize("hello", ["hi", "Hi!", "hey atlas", "Hey Atlas", "hello", "good morning"])
def test_a_greeting_is_one_short_line_with_an_offer(hello):
    qa, rt = box(Recorder("It's Sunday at 9 AM, I'm an AI and I've been running for a minute."))
    said = ask(qa, hello)
    assert said == "Hey. What can I do for you?" and rt.seen == []  # the model is not asked, so it has nothing to recite
    nothing_volunteered(said)


@pytest.mark.parametrize("hello", ["how's it going?", "How are you?", "what's up", "Hey Atlas, how's it going", "how are you doing?",
                                   "How's your day?", "how is your day going", "what have you been up to?", "how are things"])
def test_hows_it_going_is_its_work_for_you_and_an_offer_with_or_without_a_model(hello):
    for runtime in (None, Recorder("Hey there, it's Sunday at 1:37 AM and I've been running for just a minute. I'm an AI living on this box.")):
        qa, rt = box(runtime)
        said = ask(qa, hello)
        assert said == "Good. I've planned your day and reminded you twice so far. What else can I do for you?"
        assert len(sentences(said)) <= 3 and said.endswith("?") and "planned your day" in said
        nothing_volunteered(said)
        assert getattr(rt, "seen", []) == []  # built in code: a 4B model is never trusted with this
    quiet, _ = box(done=[])
    assert ask(quiet, hello) == "Good, all quiet so far. What can I do for you?"


@pytest.mark.parametrize("chat", ["hi", "hey atlas", "how's it going?", "how are you?", "what's up", "tell me a joke", "do you like music?",
                                  "what's the capital of France?", "I'm tired"])
def test_the_facts_a_model_would_get_for_small_talk_hold_no_schedule_no_cameras_no_uptime(chat):
    qa, _ = box()
    facts = "\n".join(qa.talk.facts(chat))
    assert facts == "It is Sunday morning."  # the part of the day and nothing else
    system = qa.talk.messages(chat)[0].content
    assert "Answer only what was asked" in system and "Never bring up the time, how long you have been running, that you are an AI" in system
    nothing_volunteered(system.split("FACTS\n")[1].replace("Sunday", ""))


def test_facts_follow_the_topic_of_the_question():
    qa, _ = box()
    facts = lambda q: "\n".join(qa.talk.facts(q))  # noqa: E731
    assert "Their plan for today" in facts("what should I do with my afternoon?") and "cameras" not in facts("what should I do with my afternoon?")
    assert "Your last coaching note" in facts("how am I doing with my habits?") and "Their plan for today" not in facts("how am I doing with my habits?")
    assert "It is Sunday 9:00 AM." in facts("what time is it?") and "Their day" not in facts("what time is it?")
    assert "You have been running for" in facts("how long have you been running?")
    # (2) its own day: what it did for them; the cameras' count only when asked, and what they showed only when asked what it saw
    own = facts("what did you do for me today?")
    assert "What you have done for them today: I've planned your day and reminded you twice and answered six questions." in own
    assert "rehearse" not in own and "cameras" not in own
    assert "The last one was at 8:40 AM." in facts("how many moments today?") and "rehearse" not in facts("how many moments today?")
    assert "Two people rehearse a presentation." in facts("what was the last thing you saw?")


# --- (3) the calendar, said like a person ---

SCHEDULE = {
    "what's on my calendar today?": "You've got four things today. First is your COP 4533 lecture at 10:40, then your PHY 2049 lecture at 2. Want the rest?",
    "What does my day look like?": "You've got four things today. First is your COP 4533 lecture at 10:40, then your PHY 2049 lecture at 2. Want the rest?",
    "what's next?": "Next up is your COP 4533 lecture at 10:40.",
    "Hey Atlas, what's next": "Next up is your COP 4533 lecture at 10:40.",
    "do I have anything this afternoon?": "Yes, you've got your PHY 2049 lecture at 2, then the mentor check-in at 4:30 PM.",
    "when is my next class?": "Your COP 4533 lecture is at 10:40.",
    "what time is my next class, and where?": "Your COP 4533 lecture is at 10:40. It's at L3Harris Engineering Center.",
    "am I free at 3?": "No, you've got your PHY 2049 lecture then, until 3:15 PM.",
    "am I free at noon?": "Yes, you're free at noon.",
    "am I busy at 11?": "Yes, you've got your COP 4533 lecture then, until 11:30.",
    "what's on tonight?": "Tonight you've got Dinner with Maya at 7, and that's it tonight.",
    "run me through my day": "You've got four things today. First is your COP 4533 lecture at 10:40, then your PHY 2049 lecture at 2. Want the rest?",
    "what's due this week?": "You've got Project - Milestone1 Fall2026 due Tuesday at 3:55 PM.",
}
ALLOWED = {"what's next?": 1, "Hey Atlas, what's next": 1, "when is my next class?": 1, "what time is my next class, and where?": 1, "am I free at 3?": 1,
           "am I free at noon?": 0, "am I busy at 11?": 1}


@pytest.mark.parametrize("question", list(SCHEDULE))
def test_a_schedule_question_is_answered_in_sentences_never_read_out(question):
    for runtime in (None, Recorder("COP4533 lecture 10:40 AM to 11:30 AM, L3Harris Engineering Center; PHY2049 lecture ...")):
        qa, rt = box(runtime)
        said = ask(qa, question)
        assert said == SCHEDULE[question]
        assert not RECORD.search(said), said  # no ranges, brackets, source tags, camera ids, weekday stamps or semicolon lists
        assert all(s[0].isupper() and s[-1] in ".?" for s in sentences(said)) and len(sentences(said)) <= 3
        named = sum(1 for t in ("lecture at", "lecture then", "check-in", "Dinner with Maya", "Board game", "Milestone") if t in said)
        assert named <= ALLOWED.get(question, 3)  # only what was asked: one thing for "what's next?", three at most for a rundown
        assert "rehearse" not in said and "Based on" not in said
        assert getattr(rt, "seen", []) == []  # from the data: no model retells the calendar, so none can invent an event


def test_a_course_code_is_said_by_its_name_when_the_box_knows_one():
    qa, _ = box(names={"COP4533": "algorithms", "PHY2049": "physics"})
    assert ask(qa, "do I have anything this afternoon?") == "Yes, you've got your physics lecture at 2, then the mentor check-in at 4:30 PM."
    assert ask(qa, "what's next?") == "Next up is your algorithms lecture at 10:40."
    assert TalkConfig.of({"names": {"cop 4533": "algorithms"}}).names == (("COP4533", "algorithms"),)


def test_the_rest_comes_when_asked_for_and_everything_is_everything():
    qa, _ = box()
    assert ask(qa, "what does my day look like?").endswith("Want the rest?")
    rest = ask(qa, "yes")
    assert rest == "You've got your COP 4533 lecture at 10:40, your PHY 2049 lecture at 2, the mentor check-in at 4:30 PM, Dinner with Maya at 7 tonight, then Board game night at 8 tonight."
    assert not RECORD.search(rest)
    assert ask(qa, "what's everything on my calendar today?") == rest  # asked for all of it: the optional evening too
    assert qa.talk.rest is None and qa.talk.template("yes") != rest  # a later "yes" is just a yes


def test_the_jetson_transcript_campus_listings_are_not_read_out_and_not_called_yours():
    """On the box (main, real model): "Coming up today: U Can Finish Race, Sun 7:30 AM; Theatre UCF Open House, Sun 11:00 AM; ...";
    "what does my day look like?" -> "I don't know your day yet ..."; "what's next?" -> "I don't know what's next for you ..."."""
    campus = [item(f"s{i}", title, hour, minute, connector="site", url=f"https://events.ucf.edu/{i}") for i, (title, hour, minute) in enumerate([
        ("U Can Finish Race", 7, 30), ("Theatre UCF Open House", 11, 0), ("Fall Climbing Competition: Dungeons & Dynos", 12, 0),
        ("UCF Women's Soccer vs. Oklahoma State", 13, 0), ("Graduate School Mixer", 15, 0)])]
    night = datetime(2026, 10, 11, 5, 37, tzinfo=UTC)  # Sunday 1:37 AM
    for question in ("What's on my calendar today?", "What does my day look like?"):
        qa, rt = box(Recorder("I don't know your day yet, but I can see you have classes and assignments on Canvas waiting for you."), items=campus)
        qa.clock = qa.talk.clock = lambda: night
        a = asyncio.run(qa.ask(question))
        assert a.text == "There are five things on campus today. First is U Can Finish Race at 7:30, then Theatre UCF Open House at 11. Want the rest?"
        assert len(a.citations) == 5 and rt.seen == []  # the same five items behind it, cited as before
    qa, rt = box(Recorder("I don't know what's next for you."), items=campus)
    qa.clock = qa.talk.clock = lambda: night
    qa.talk.day_facts = None  # no Day plan on that box; with one, "what's next?" is the narrator's spoken words (below)
    assert asyncio.run(qa.ask("What's next?")).text == "Nothing else on your calendar today."  # a listing is not their next thing
    assert asyncio.run(qa.ask("do I have anything this afternoon?")).text == (
        "On campus this afternoon there's Dungeons & Dynos at noon, UCF Women's Soccer vs. Oklahoma State at 1, then Graduate School Mixer at 3.")


def test_an_empty_calendar_is_said_plainly():
    qa, _ = box(items=[])
    assert ask(qa, "what does my day look like?") == "You're on plan: your study block runs until twenty past ten. The lecture is at twenty to eleven."
    qa.talk.day_facts = None
    assert ask(qa, "what does my day look like?") == "Nothing on your calendar today."
    assert ask(qa, "what's on my calendar today?") == "Nothing on your calendar today."
    assert ask(qa, "anything due tomorrow?") == "Nothing due tomorrow."


def test_how_am_i_doing_is_the_narrators_spoken_words_not_the_screens():
    qa, _ = box()
    assert ask(qa, "how am I doing?") == "You're on plan: your study block runs until twenty past ten."


# --- routing is untouched: only the words changed ---

def test_a_complex_request_still_takes_the_tool_loop_and_its_full_answer_comes_back_unchanged():
    long_answer = ("While you were out a courier left a box at the door around nine, and two people rehearsed a presentation at the desk. "
                   "For this afternoon I'd start with the physics lecture at 2, keep the mentor check-in at 4:30, and use the gap before dinner "
                   "for the milestone writeup, since it's due Tuesday. Want me to block that out?")
    call = ChatResult(content="", model_version="v1", tool_calls=[ToolCall(id="t1", name="search_memory", arguments={"q": "courier door"}),
                                                                  ToolCall(id="t2", name="upcoming", arguments={"days": 1})])
    rt = FakeRuntime(chat_replies=[call, ChatResult(content=long_answer, model_version="v1")])
    qa, _ = box(rt)
    question = "what happened while I was out, and what should I do about my afternoon?"
    assert qa.prefetch(question) is None and not qa.talk.wants(question)  # no fixed path, not small talk: the tool loop, as on main
    a = asyncio.run(qa.ask(question))
    assert a.text == long_answer and len(sentences(a.text)) == 3  # nothing shortened, nothing reworded
    assert [c[0] for c in rt.calls] == ["chat", "chat"]  # the tool-choice round, then the answer from what the tools found
    assert "Use tools to find evidence." in rt.calls[0][1][0].content and rt.calls[1][1][-1].role == "tool"
    assert a.citations  # what the tools found is still cited


@pytest.mark.parametrize("question,path", [
    ("Where did I leave my keys?", "prefetch"), ("What happened overnight?", "prefetch"), ("did anyone come by?", "tool loop"),
    ("is the stove on?", "tool loop"), ("do I have any homework?", "tool loop"), ("what do students say about COP 3502?", "tool loop"),
    ("tell me a joke", "talk"), ("explain recursion simply", "talk"), ("I'm stressed about my exam", "talk"),
])
def test_questions_that_are_not_about_the_calendar_go_where_they_always_went(question, path):
    qa, _ = box(items=[])
    fixed = qa.prefetch(question) is not None
    assert (path == "prefetch") == fixed and (path == "talk") == (not fixed and qa.talk.wants(question))


def test_a_camera_question_still_gets_the_no_evidence_line():
    qa, _ = box(items=[])
    assert asyncio.run(qa.ask("Where is my wallet?")).text == NO_EVIDENCE


# --- what it did for you is counted from real events ---

def event(type, minutes_ago, **fields):
    return SimpleNamespace(type=type, ts=NOW - timedelta(minutes=minutes_ago), **fields)


def test_done_for_you_is_counted_from_the_log_most_significant_first():
    done = DoneToday(QA(FakeRuntime(), FakeMemory(), clock=lambda: NOW).sleep.tz)
    assert done.clauses(NOW) == []
    for e in (event("day.plan.proposed", 60), event("habits.nudge.raised", 40), event("habits.nudge.raised", 30),
              event("agent.answer.generated", 20), event("agent.action.proposed", 15, action_id="a1", kind="send_message"),
              event("experience.approval.decided", 14, action_id="a1"), event("workspace.action.completed", 14, action_id="a1", status="done"),
              event("agent.action.proposed", 2, action_id="a2", kind="connector_write"),
              event("day.plan.proposed", 60 * 30), event("agent.action.proposed", 60 * 30, action_id="old", kind="browse")):
        done.on_event(e)
    assert done.clauses(NOW) == ["planned your day", "left one thing waiting for your OK", "sent the message you approved", "reminded you twice",
                                 "answered one question"]
    done.on_event(event("day.replan.proposed", 1))
    done.on_event(event("workspace.action.completed", 1, action_id="a2", status="denied"))
    assert done.clauses(NOW)[:2] == ["re-planned your day", "sent the message you approved"]  # denied: nothing is waiting, nothing was done
    assert done.clauses(NOW + timedelta(minutes=30))[0] == "re-planned your day"
    assert done.clauses(NOW + timedelta(days=1)) == []  # a new day starts clean
