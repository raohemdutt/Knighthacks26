# OWNER'S RULE (Hemdutt): by voice Atlas says the answer once, like a person: the calendar as sentences, "how's it going?"
# as its work for you and an offer, and never "Based on N pieces of evidence" after a calendar answer.
# The box's voice posts what it heard straight to /api/ask(/stream): these run that whole path with the real QA.
import asyncio
import re

from contexts.agent.tests.test_natural_speech import RUNDOWN, box
from contexts.voice.service import evidence_count
from contexts.voice.session import VoiceSession
from contexts.voice.tests.fakes import FakeMic, FakeSpeaker, FakeStt, FakeTts, make_config
from contexts.voice.text import SentenceStream, clean_for_speech, speech_for_answer


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
    assert answer == RUNDOWN
    # each sentence once, in order, exactly as written: a bare morning "10:40" is said as it is, with no "AM" guessed onto it
    assert " ".join(s.tts.said) == answer and "10:40 AM" not in " ".join(s.tts.said)
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


def test_a_whole_day_walked_through_is_spoken_to_the_end_not_cut_off():
    """The owner's rule: "run me through my day" is the WHOLE day. At the old 420-character cap the voice stopped before the
    campus clause and the closing offer, and on a fuller day it would have dropped the person's own commitments."""
    from contexts.agent.tests.test_natural_speech_2 import WHOLE_DAY

    said = WHOLE_DAY.replace("&", "and")  # the one thing the voice rewords in it
    assert len(WHOLE_DAY) > 420 and speech_for_answer(WHOLE_DAY) == said
    stream = SentenceStream()
    pieces = [s for i in range(0, len(WHOLE_DAY), 9) for s in stream.feed(WHOLE_DAY[i:i + 9])] + [stream.flush()]
    assert " ".join(clean_for_speech(p) for p in pieces) == said


def test_a_bare_time_is_spoken_as_written_and_only_a_24_hour_clock_is_reworded():
    for said in ("Next up is your COP 4533 lecture at 10:40.", "Leave by 10:30 for that one.", "You're free at 2.", "It starts at 9:15, then lunch at noon.",
                 "The mentor check-in is at 4:30 PM."):
        assert clean_for_speech(said) == said
    assert clean_for_speech("It was at 15:40.") == "It was at 3:40 PM." and clean_for_speech("At 07:05:00 someone came.") == "At 7:05 AM someone came."
    assert clean_for_speech("At 0:15 and 13:05.") == "At 12:15 AM and 1:05 PM."


def test_a_markdown_list_from_the_tool_loop_is_spoken_as_sentences_with_no_asterisks_or_bullets():
    """On the box the complex answer came back as "- **Study the ...** immediately after ...": the text is the tool loop's and
    is not changed; what is said is each item as its own sentence."""
    answer = ("Here's the plan:\n- **Study the algorithms notes** immediately after your 10:40 lecture.\n- **Eat lunch** at noon\n"
              "1. **Start the milestone writeup** at 2, since it's due Tuesday at 3:55 PM.")
    said = ("Here's the plan: Study the algorithms notes immediately after your 10:40 lecture. Eat lunch at noon. "
            "Start the milestone writeup at 2, since it's due Tuesday at 3:55 PM.")
    assert clean_for_speech(answer) == said and speech_for_answer(answer) == said
    stream = SentenceStream()  # and the same when it arrives piece by piece, as the box's voice gets it
    pieces = [s for i in range(0, len(answer), 7) for s in stream.feed(answer[i:i + 7])] + [stream.flush()]
    spoken = " ".join(clean_for_speech(p) for p in pieces if clean_for_speech(p))
    assert spoken == said and not re.search(r"[*_#`]|^\s*[-•]|\s[-•]\s", spoken)


def test_only_what_the_cameras_saw_is_announced_as_evidence():
    assert evidence_count([{"url": "https://canvas/m1", "title": "Milestone"}, {"title": "COP4533 lecture", "url": None}]) == 0
    assert evidence_count([{"frame_ref": "frames:cam-desk:1"}, {"episode_id": "e1"}, {"url": "https://x.test"}]) == 2
    assert evidence_count([]) == 0 and evidence_count(None) == 0
