"""A conversation on the box (`follow_up`): after an answer the microphone opens again for a reply, with no "Hey Atlas"."""

import threading

import pytest

from contexts.voice.session import SAY_AGAIN
from contexts.voice.tests.fakes import FakeMic, FakeSpeaker, FakeStt, question, silence
from contexts.voice.tests.test_session import WAKE, _close_sessions, build, until  # noqa: F401  (the fixture closes every session)

ANSWER = "You left them on the desk at 3:40 PM. Based on 2 pieces of evidence."  # what build()'s ask is spoken as
QUIET = silence(100)  # three seconds of nobody speaking: longer than the window below


def converse(heard: list[str], opens: list[list[bytes]] | None = None, **cfg):
    """A tap, then the room says `heard` one after the other (the first is the question), then stays quiet."""
    mic = FakeMic(opens=opens or [question()] * len(heard) + [QUIET])
    return build(mic=mic, stt=FakeStt(script=list(heard), text="not reached"), **{"follow_up": True, "follow_up_wait_s": 1.0, **cfg})


def test_a_conversation_from_hey_atlas_to_silence_and_back_to_the_wake_listener():
    mic = FakeMic(opens=[question(), question(), question(), QUIET, silence(5)])  # wake, question, reply, nobody, the room again
    s, mic, speaker, asked = build(mic=mic, stt=FakeStt(script=["Hey Atlas.", "where are my keys", "and my wallet?"], text="[BLANK_AUDIO]"),
                                   follow_up=True, follow_up_wait_s=1.0, **WAKE)
    s.start_wake()
    until(lambda: asked == ["where are my keys", "and my wallet?"] and s.state == "idle")  # the reply needed no "Hey Atlas"
    assert s.tts.said == ["What's up?", ANSWER, ANSWER] and s.heard == "and my wallet?" and s.problem == ""
    assert s.stt.quick[:3] == [True, False, False]  # the reply is heard with the careful model, like the question
    assert not speaker.mic_was_open_while_playing
    until(lambda: mic.opened >= 6 and not s._thread.is_alive())  # the wake listener has the microphone again
    assert s._wake_thread.is_alive() and s.status()["follow_up"] is False


def test_follow_up_off_is_one_turn_per_tap():
    s, mic, speaker, asked = build(follow_up=False)  # the same room keeps asking: only the first is heard
    s.talk()
    s.wait()
    assert asked == ["where did I leave my keys"] and mic.opened == 1 and len(speaker.played) == 1 and s.status()["follow_up"] is False


def test_a_quiet_room_after_the_answer_ends_like_a_single_turn():
    s, mic, speaker, asked = converse(["where did I leave my keys"])
    s.talk()
    s.wait()
    assert asked == ["where did I leave my keys"] and s.state == "idle" and s.problem == ""
    assert s.heard == "where did I leave my keys" and len(speaker.played) == 1 and mic.opened == 2
    assert not speaker.mic_was_open_while_playing


def test_the_window_is_its_own_state_on_the_screen_and_a_blip_opens_it():
    mic = FakeMic(opens=[question(), silence(20)], endless=True)
    s, mic, speaker, asked = build(mic=mic, follow_up=True, follow_up_wait_s=60.0, chime=True)
    s.talk()
    until(lambda: s.status()["follow_up"])
    assert s.state == "listening" and len(speaker.played) == 3  # heard-it blip, the answer, your-turn blip
    assert speaker.played[0] == speaker.played[2] and not speaker.mic_was_open_while_playing
    s.talk()  # a tap with nothing said ends the conversation
    s.wait()
    assert s.state == "idle" and len(asked) == 1 and s.problem == ""


def test_its_own_voice_is_not_a_reply():
    s, mic, _, asked = converse(["where are my keys", "You left them on the desk at 3:40 PM."])
    s.talk()
    s.wait()
    assert asked == ["where are my keys"] and s.tts.said == [ANSWER] and s.heard == "where are my keys" and s.problem == ""


@pytest.mark.parametrize("noise", ["", "Oh."])  # "": whisper was not sure, or wrote "Thank you." / a music note for a noise
def test_noise_is_dropped_without_a_word(noise):
    s, mic, _, asked = converse(["where are my keys", noise])
    s.talk()
    s.wait()
    assert asked == ["where are my keys"] and SAY_AGAIN not in s.tts.said and s.problem == "" and s.heard == "where are my keys"


def test_a_real_reply_after_one_noise_is_still_answered():
    s, _, _, asked = converse(["where are my keys", "", "Hey Atlas, why there?"], wake_word=True)
    s.talk()
    s.wait()
    assert asked == ["where are my keys", "why there?"]  # and "Hey Atlas" inside the window is not part of it


def test_two_unclear_sounds_in_a_row_end_the_conversation():
    s, mic, _, asked = converse(["where are my keys", "", "", "why there?"], opens=[question()])  # the room never goes quiet
    s.talk()
    s.wait()
    assert asked == ["where are my keys"] and mic.opened == 3 and s.state == "idle"


def test_a_noise_does_not_win_more_time():
    now = [0.0]
    s, mic, _, asked = converse(["where are my keys", "", "why there?"], opens=[question()], follow_up_wait_s=5.0)
    s.clock = lambda: now[0]
    transcribe = s.stt.transcribe

    def slow(pcm, quick=False):  # the noise used up the window
        now[0] += 6.0
        return transcribe(pcm, quick)

    s.stt.transcribe = slow
    s.talk()
    s.wait()
    assert asked == ["where are my keys"] and mic.opened == 2


@pytest.mark.parametrize("said,line", [("Thanks.", "Any time."), ("Never mind.", "Okay.")])
def test_a_goodbye_gets_one_short_line_and_no_question_is_asked(said, line):
    s, mic, _, asked = converse(["where are my keys", said], opens=[question()])
    s.talk()
    s.wait()
    assert asked == ["where are my keys"] and s.tts.said == [ANSWER, line] and s.answer == line and mic.opened == 2


def test_a_conversation_stops_at_the_most_turns():
    s, mic, _, asked = converse(["one", "two", "three", "four", "five"], opens=[question()], follow_up_max_turns=3)
    s.talk()
    s.wait()
    assert asked == ["one", "two", "three"] and mic.opened == 3 and s.state == "idle"


def test_an_answer_cut_short_by_a_tap_does_not_open_the_microphone():
    s, mic, speaker, asked = converse(["where are my keys"], opens=[question()])
    speaker.block = threading.Event()
    s.talk()
    until(lambda: s.state == "speaking")
    s.talk()
    s.wait()
    assert s.state == "idle" and mic.opened == 1 and len(asked) == 1


def test_an_answer_that_could_not_be_spoken_does_not_open_the_microphone():
    mic = FakeMic()
    s, mic, _, asked = build(mic=mic, speaker=FakeSpeaker(mic, missing=True), follow_up=True)
    s.talk()
    s.wait()
    assert mic.opened == 1 and len(asked) == 1 and "Couldn't speak the answer" in s.problem


def test_a_streamed_answer_is_followed_by_a_reply_too():
    s, mic, _, _ = converse(["where are my keys", "on the desk by the", "why there?"])
    asked = []

    def ask_stream(text, on_piece):
        asked.append(text)
        on_piece("They are on the desk by the window. ")
        return "They are on the desk by the window.", 0

    s.ask_stream = ask_stream
    s.talk()
    s.wait()
    assert asked == ["where are my keys", "why there?"]  # the middle one was the speaker, heard by the microphone


def test_stopping_during_the_window_ends_cleanly():
    mic = FakeMic(opens=[question(), question(), silence(20)], endless=True)
    s, mic, _, asked = build(mic=mic, stt=FakeStt(script=["Hey Atlas."]), follow_up=True, follow_up_wait_s=60.0, **WAKE)
    s.start_wake()
    until(lambda: s.status()["follow_up"])
    s.close()
    s.wait()
    assert not s._thread.is_alive() and not s._wake_thread.is_alive() and not mic.is_open and s.state == "idle"
    assert len(asked) == 1  # this session's own threads are checked above; the process may hold other tests' threads


def test_turning_the_voice_off_during_the_window_ends_it():
    mic = FakeMic(opens=[question(), silence(20)], endless=True)
    s, mic, _, _ = build(mic=mic, follow_up=True, follow_up_wait_s=60.0)
    s.talk()
    until(lambda: s.status()["follow_up"])
    assert s.set_enabled(False)["state"] == "off"
    s.wait()
    assert not mic.is_open and not s._conversing


def test_a_turn_that_raises_gives_the_microphone_back_to_the_wake_listener():
    mic = FakeMic(opens=[question(), question(), silence(5)])
    s, mic, _, asked = build(mic=mic, stt=FakeStt(script=["Hey Atlas.", "what time is it"], text="[BLANK_AUDIO]"), follow_up=True, **WAKE)

    def boom():
        raise RuntimeError("window broke")

    s._listen_for_reply = boom
    s.start_wake()
    until(lambda: "window broke" in s.problem and s.state == "idle")
    until(lambda: mic.opened >= 4)  # listening for "Hey Atlas" again
    assert s._wake_thread.is_alive() and not s._conversing and len(asked) == 1


# -- lines the box says on its own (reminders and check-ins from contexts/day, through say()) ---------------------------------

def test_a_line_that_arrives_during_the_window_waits_until_it_closes():
    mic = FakeMic(opens=[question(), silence(20)], endless=True)
    s, mic, speaker, asked = build(mic=mic, follow_up=True, follow_up_wait_s=60.0)
    s.talk()
    until(lambda: s.status()["follow_up"])
    assert s.say("Time to leave for class.")["state"] == "listening"  # not spoken over the open microphone
    assert s.tts.said == [ANSWER]
    mic.endless = False  # nobody replies: the window closes
    s.wait()
    assert s.tts.said == [ANSWER, "Time to leave for class."] and len(asked) == 1 and s.state == "idle"
    assert not speaker.mic_was_open_while_playing and mic.opened == 2  # and the line opens no window of its own


def test_a_line_that_arrives_while_a_reply_is_answered_ends_the_conversation_after_that_answer():
    s, mic, speaker, _ = converse(["where are my keys", "why there?", "and then?"], opens=[question()])
    asked, thinking, go = [], threading.Event(), threading.Event()

    def ask(text):
        asked.append(text)
        if len(asked) == 2:  # the reply's answer is held back until the line has arrived
            thinking.set()
            go.wait(3)
        return "Because you put them there.", 0

    s.ask = ask
    s.talk()
    assert thinking.wait(3)
    s.say("Time to leave for class.")
    go.set()
    s.wait()
    assert asked == ["where are my keys", "why there?"] and s.tts.said[-2:] == ["Because you put them there.", "Time to leave for class."]


def test_a_line_the_box_says_on_its_own_opens_no_window():
    s, mic, speaker, asked = build(follow_up=True)
    s.say("You have a lecture at 10:40. Ready?")
    s.wait()
    assert mic.opened == 0 and asked == [] and len(speaker.played) == 1 and s.state == "idle"
