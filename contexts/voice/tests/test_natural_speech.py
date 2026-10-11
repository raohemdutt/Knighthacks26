# OWNER'S RULE (Hemdutt): by voice Atlas says the answer once, like a person: the calendar as sentences, "how's it going?"
# as its work for you and an offer, and never "Based on N pieces of evidence" after a calendar answer.
# The box's voice posts what it heard straight to /api/ask(/stream): these run that whole path with the real QA.
import asyncio
import re

from contexts.agent.tests.test_natural_speech import box
from contexts.voice.service import evidence_count
from contexts.voice.session import VoiceSession
from contexts.voice.tests.fakes import FakeMic, FakeSpeaker, FakeStt, FakeTts, make_config


def heard(question: str, **kw) -> tuple[VoiceSession, str]:
    """One spoken turn: `question` is what whisper heard; the answer comes from QA.ask, streamed as the gateway streams it."""
    qa, _ = box(**kw)
    answers = []

    def ask_stream(text, on_piece):
        a = asyncio.run(qa.ask(text, stream=on_piece))
        answers.append(a.text)
        return a.text, evidence_count([c.model_dump() for c in a.citations])

    mic = FakeMic()
    s = VoiceSession(make_config(), mic, FakeSpeaker(mic), FakeStt(text=question), FakeTts(), lambda text: ("", 0), ask_stream=ask_stream)
    s.talk()
    s.wait()
    s.close()
    return s, answers[0]


def test_the_voice_says_the_calendar_answer_once_as_sentences():
    s, answer = heard("what does my day look like")
    assert answer == "You've got four things today. First is your COP 4533 lecture at 10:40, then your PHY 2049 lecture at 2. Want the rest?"
    # each sentence once, in order; the voice writes a morning "10:40" as "10:40 AM" and leaves the rest alone
    assert " ".join(s.tts.said) == answer.replace("at 10:40,", "at 10:40 AM,")
    assert s.answer == answer and s.problem == ""
    said = " ".join(s.tts.said)
    assert "Based on" not in said and not re.search(r"[AP]M to|;|\[|\(|Sun \d|Coming up", said)


def test_the_voice_says_hows_it_going_once_and_nothing_else():
    for question in ("how's it going", "Hey Atlas, how are you?"):
        s, answer = heard(question)
        assert answer == "Good. I've planned your day and reminded you twice so far. What else can I do for you?"
        assert " ".join(s.tts.said) == answer  # exactly that text: no second answer, no evidence line
    s, answer = heard("hi")
    assert s.tts.said == ["Hey. What can I do for you?"]


def test_an_afternoon_time_keeps_its_pm_so_the_voice_cannot_call_it_morning():
    s, answer = heard("do I have anything this afternoon")
    assert " ".join(s.tts.said) == answer == "Yes, you've got your PHY 2049 lecture at 2, then the mentor check-in at 4:30 PM."


def test_only_what_the_cameras_saw_is_announced_as_evidence():
    assert evidence_count([{"url": "https://canvas/m1", "title": "Milestone"}, {"title": "COP4533 lecture", "url": None}]) == 0
    assert evidence_count([{"frame_ref": "frames:cam-desk:1"}, {"episode_id": "e1"}, {"url": "https://x.test"}]) == 2
    assert evidence_count([]) == 0 and evidence_count(None) == 0
