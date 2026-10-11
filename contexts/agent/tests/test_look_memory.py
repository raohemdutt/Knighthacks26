"""The look memory's three questions: answered through a port from what the Day loop remembered, never guessed."""

import asyncio

from contexts.agent.application.qa import QA
from contexts.agent.application.talk import LOOK_ASKED, Talk, normal
from contexts.agent.tests.test_talk import MEMORY, NOW
from runtime import FakeRuntime

SAID = {"last": "The last time you looked this strained was Friday afternoon, when you'd gone two hours without moving.",
        "why": "I can't know how you feel, only what I saw: your face has read as strained for a while.",
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
