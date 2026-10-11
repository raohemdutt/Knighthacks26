"""The look memory's three questions: answered through a port from what the Day loop remembered, never guessed."""

import asyncio
from datetime import timedelta

from contexts.agent.application.qa import QA
from contexts.agent.application.talk import LOOK_ASKED, Talk, normal
from contexts.agent.tests.test_talk import MEMORY, NOW
from runtime import FakeRuntime

SAID = {"last": "The last time you looked this tired was Friday afternoon, when you'd gone two hours without moving.",
        "why": "I can't know how you feel, only what I saw: your face has looked tired for a while.",
        "helped": "Last time, Friday afternoon, ten minutes away from the desk helped: the look eased within half an hour."}
ASKED = {
    "last": ["When did I last look this tired?", "Atlas, when did I last look this tired", "when was the last time I looked this strained?",
             "have I looked this tired before?", "hey Atlas, when did I look so worn out?"],
    "why": ["why do you think I'm tired?", "Why do you think I am tired", "why do you say I look tired?", "why do I look so tired?",
            "what makes you think I'm tired?"],
    "helped": ["what helped last time?", "What helped me last time", "what worked last time?", "what helped before?"],
}


def qa(port=SAID.get, replies=()):
    rt = FakeRuntime(chat_replies=list(replies))
    q = QA(rt, MEMORY, clock=lambda: NOW)
    q.talk = Talk(rt, q.clock, memory=MEMORY, tz=q.sleep.tz)
    q.talk.look_memory = port
    return q, rt


def test_each_way_of_asking_gets_the_memorys_own_sentence():
    q, _ = qa()
    for kind, questions in ASKED.items():
        for question in questions:
            heard = []
            a = asyncio.run(q.ask(question, stream=heard.append))
            assert a.text == SAID[kind] == "".join(heard), question
            assert a.citations == []  # no picture is behind it
    assert q.talk.turns[-1] == (ASKED["helped"][-1], SAID["helped"])  # a turn of the conversation, like any other


def test_nothing_else_is_taken_and_two_questions_in_one_are_left_to_the_usual_path():
    asked = []
    q, _ = qa(port=lambda kind: asked.append(kind) or SAID[kind])
    for other in ("I'm tired", "what helped", "when did I last see my keys?", "why is the sky blue?", "what did I do last time?",
                  "where are my keys?", "what happened today?", "am I free at 3?", "how did I sleep?",
                  "why do you think I'm tired, and what should I do about my exam?"):
        assert q.talk.look(other) is None, other
    assert asked == []
    assert not [k for k, rx in LOOK_ASKED.items() if rx.search(normal("where did I last see my tired old laptop?"))]


def test_a_box_that_remembers_no_looks_answers_as_it_did_before():
    with_port, _ = qa(port=lambda kind: None)
    without, _ = qa(port=None)
    for q in (with_port, without):
        assert q.talk.look("what helped last time?") is None and q.talk.answer_look("what helped last time?") is None
    a, b = asyncio.run(with_port.ask("what helped last time?")), asyncio.run(without.ask("what helped last time?"))
    assert a.text == b.text


def test_a_port_that_fails_never_fails_the_answer():
    def broken(kind):
        raise RuntimeError("no day loop")

    q, _ = qa(port=broken)
    assert q.talk.look("when did I last look this tired?") is None
    assert asyncio.run(q.ask("why do you think I'm tired?")).text


# ---------- stage two: what the box has learned from the looks it remembers ----------

LEARNED = {"helps": "Getting up, from what I've seen. The look eased five of the six times you got up, and eased neither of the two times you only had a drink.",
           "causes": "I can't say what causes it, only what the looks share: eleven of the thirteen looks I remember started after ninety minutes or more at the desk.",
           "how": "From what I saw on this box, nothing more: getting up eased it five of six times, and water alone none of two. "
                  "It's a count over the thirteen looks I remember, not a reading of how you feel."}
LEARNED_ASKED = {
    "helps": ["What usually helps when I look tired?", "what helps when I look this tired?", "Atlas, what usually helps me when I'm tired?",
              "what normally works when I look strained?", "what helps the most when I look worn out?"],
    "causes": ["What makes me look tired?", "what usually makes me look so tired?", "what causes me to look tired?", "when do I usually look tired?",
               "what brings that look on?"],
    "how": ["how do you know what helps?", "How do you know what usually helps me?", "how do you know what makes me look tired?",
            "how do you know I look tired?"],
}


def test_what_it_learned_is_asked_for_in_plain_words_and_answered_with_the_memorys_own_sentence():
    q, rt = qa(port=(SAID | LEARNED).get)
    for kind, questions in LEARNED_ASKED.items():
        for question in questions:
            heard = []
            a = asyncio.run(q.ask(question, stream=heard.append))
            assert a.text == LEARNED[kind] == "".join(heard) and a.citations == [], question


def test_how_do_you_know_is_about_the_looks_only_right_after_an_answer_about_them():
    now = {"t": NOW}
    asked = []
    rt = FakeRuntime(chat_replies=[])
    q = QA(rt, MEMORY, clock=lambda: now["t"])
    q.talk = Talk(rt, q.clock, memory=MEMORY, tz=q.sleep.tz)
    q.talk.look_memory = lambda kind: asked.append(kind) or (SAID | LEARNED)[kind]
    # on its own "how do you know?" could be about anything: it is not the look memory's
    for other in ("how do you know?", "what usually helps?", "what brings it on?", "says who?"):
        assert q.talk.look(other) is None, other
    assert asked == []
    assert q.talk.look("what usually helps when I look tired?") == LEARNED["helps"]
    assert q.talk.look("How do you know?") == LEARNED["how"] and q.talk.look("and what brings it on?") == LEARNED["causes"]
    assert q.talk.look("how did you work that out?") == LEARNED["how"] and q.talk.look("so what usually helps?") == LEARNED["helps"]
    assert q.talk.look("how do you know my keys are there?") is None  # a follow-up about something else is still not taken
    now["t"] = NOW + timedelta(minutes=10)  # the conversation moved on: the same words are anybody's again
    assert q.talk.look("how do you know?") is None
    assert q.talk.look("what makes me look tired?") == LEARNED["causes"]
    q.talk.forget()
    assert q.talk.look("how do you know?") is None  # ... and a forgotten conversation has no follow-up


def test_the_new_questions_take_nothing_else_and_a_box_with_no_look_memory_answers_as_before():
    asked = []
    q, _ = qa(port=lambda kind: asked.append(kind) or LEARNED.get(kind))
    for other in ("what helps with a headache?", "what usually helps?", "what makes me late?", "how do you know my schedule?", "what makes bread rise?",
                  "when do I usually leave for class?", "what works when I'm stuck on homework?", "how do you know where my keys are?",
                  "what usually helps when I look tired, and what's on my calendar?"):
        assert q.talk.look(other) is None, other
    assert asked == []
    none, _ = qa(port=lambda kind: None)
    without, _ = qa(port=None)
    for question in ("what usually helps when I look tired?", "what makes me look tired?", "how do you know what helps?"):
        assert none.talk.look(question) is None and without.talk.look(question) is None
        assert asyncio.run(none.ask(question)).text == asyncio.run(without.ask(question)).text
