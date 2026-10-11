import threading
import time

import pytest

from contexts.voice.session import SAY_AGAIN, VoiceSession
from contexts.voice.tests.fakes import FakeMic, FakeSpeaker, FakeStt, FakeTts, make_config, question, silence, tone


def build(mic=None, speaker=None, stt=None, tts=None, ask=None, **cfg):
    mic = mic or FakeMic()
    speaker = speaker or FakeSpeaker(mic)
    asked = []

    def default_ask(text):
        asked.append(text)
        return "You left them on the desk at 15:40. See frames:cam-desk:1760000000000.", 2

    s = VoiceSession(make_config(**cfg), mic, speaker, stt or FakeStt(), tts or FakeTts(), ask or default_ask)
    return s, mic, speaker, asked


@pytest.fixture(autouse=True)
def _close_sessions(monkeypatch):
    """Every session built in a test is closed after it, so no wake listener outlives its test."""
    sessions = []
    original = VoiceSession.__init__

    def tracking_init(self, *a, **kw):
        original(self, *a, **kw)
        sessions.append(self)

    monkeypatch.setattr(VoiceSession, "__init__", tracking_init)
    yield
    for s in sessions:
        s.close()


def until(condition, seconds=3.0):
    deadline = time.time() + seconds
    while not condition() and time.time() < deadline:
        time.sleep(0.01)
    assert condition()


def wait_for(s, state, seconds=3.0):
    deadline = time.time() + seconds
    while s.state != state and time.time() < deadline:
        time.sleep(0.01)
    assert s.state == state


def test_a_question_goes_through_the_ask_path_and_the_answer_is_spoken():
    s, mic, speaker, asked = build()
    assert s.talk()["state"] == "listening"
    s.wait()
    assert asked == ["where did I leave my keys"]  # only text goes on; the audio does not
    assert s.state == "idle" and s.problem == ""
    assert s.heard == "where did I leave my keys"
    assert len(speaker.played) == 1 and speaker.played[0][1] == 22050
    assert s.tts.said == ["You left them on the desk at 3:40 PM. Based on 2 pieces of evidence."]  # no frame ref read out


def test_the_microphone_is_shut_while_the_speaker_plays():
    s, mic, speaker, _ = build()
    s.talk()
    s.wait()
    assert speaker.played and not speaker.mic_was_open_while_playing
    assert mic.opened == 1 and not mic.is_open


def test_a_tap_while_speaking_stops_the_speech_and_does_not_start_listening():
    mic = FakeMic()
    speaker = FakeSpeaker(mic)
    speaker.block = threading.Event()  # the "speech" lasts until released
    s, *_ = build(mic=mic, speaker=speaker)
    s.talk()
    wait_for(s, "speaking")
    s.talk()  # the stop
    s.wait()
    assert s.state == "idle" and mic.opened == 1  # it never listened again


def test_a_tap_while_thinking_is_ignored():
    release = threading.Event()

    def slow_ask(text):
        release.wait(3)
        return "Done.", 0

    s, mic, *_ = build(ask=slow_ask)
    s.talk()
    wait_for(s, "thinking")
    assert s.talk()["state"] == "thinking" and mic.opened == 1
    release.set()
    s.wait()
    assert s.state == "idle"


def test_a_second_tap_while_listening_ends_the_question():
    mic = FakeMic(silence(12) + tone(20), endless=True)  # speech that never pauses on its own
    s, *_ = build(mic=mic, end_silence_s=60.0, max_seconds=600.0, wait_for_speech_s=60.0)
    s.talk()
    time.sleep(0.3)
    assert s.state == "listening"
    s.talk()
    s.wait()
    assert s.state == "idle" and s.heard == "where did I leave my keys"


def test_no_microphone_is_a_message_not_a_crash():
    s, *_ = build(mic=FakeMic(missing=True))
    assert s.status()["mic"]["ok"] is False
    s.talk()
    s.wait()
    assert s.state == "idle" and s.problem.startswith("No microphone")


def test_no_speaker_leaves_the_answer_on_screen():
    mic = FakeMic()
    s, *_ = build(mic=mic, speaker=FakeSpeaker(mic, missing=True))
    s.talk()
    s.wait()
    assert s.state == "idle" and s.answer.startswith("You left them")
    assert "on screen" in s.problem
    assert s.status()["speaker"]["ok"] is False


def test_a_speaker_that_is_not_there_is_marked_absent_and_a_broken_one_is_not():
    """The operator card says "No speaker" quietly for an absent speaker, and keeps a real fault in yellow."""
    away = build(speaker=FakeSpeaker(missing=True))[0].status()["speaker"]
    assert away["ok"] is False and away["absent"] is True
    broken = build(speaker=FakeSpeaker(broken=True))[0].status()["speaker"]
    assert broken["ok"] is False and "absent" not in broken
    assert "absent" not in build()[0].status()["speaker"]


def test_listen_only_keeps_what_was_heard_and_asks_nothing():
    s, mic, speaker, asked = build()
    assert s.talk(ask=False)["state"] == "listening"
    s.wait()
    assert s.heard == "where did I leave my keys" and s.state == "idle" and s.problem == ""
    assert asked == [] and speaker.played == []  # the screen asks and speaks; the box only listened
    s._quiet_until = 0.0  # past the short pause after a turn
    s.talk()  # the ordinary button still asks and speaks
    s.wait()
    assert asked == ["where did I leave my keys"] and len(speaker.played) == 1


def test_say_speaks_the_text_on_the_box_speaker_without_listening():
    s, mic, speaker, asked = build()
    assert s.say("Your keys are on the desk.")["state"] == "thinking"
    s.wait()
    assert s.state == "idle" and s.problem == ""
    assert s.tts.said == ["Your keys are on the desk."] and len(speaker.played) == 1
    assert mic.opened == 0 and asked == []  # it only speaks; nothing is asked or heard


def test_say_is_ignored_while_the_box_is_busy_and_when_the_voice_is_off():
    mic = FakeMic()
    speaker = FakeSpeaker(mic)
    speaker.block = threading.Event()
    s, *_ = build(mic=mic, speaker=speaker)
    s.say("First.")
    wait_for(s, "speaking")
    s.say("Second.")  # already speaking: not queued over it
    speaker.block.set()
    s.wait()
    assert s.tts.said == ["First."]
    s.set_enabled(False)
    s.say("Nothing.")
    assert s.tts.said == ["First."]


def test_say_without_a_speaker_is_a_message_not_a_crash():
    mic = FakeMic()
    s, *_ = build(mic=mic, speaker=FakeSpeaker(mic, missing=True))
    s.say("Hello.")
    s.wait()
    assert s.state == "idle" and "Couldn't speak" in s.problem


def test_silence_is_not_sent_anywhere():
    s, _, _, asked = build(mic=FakeMic(silence(100)))
    s.talk()
    s.wait()
    assert asked == [] and s.stt.heard == [] and "Didn't hear" in s.problem


def test_a_blank_transcript_is_not_asked():
    s, _, _, asked = build(stt=FakeStt(text=""))
    s.talk()
    s.wait()
    assert asked == [] and "Didn't catch" in s.problem
    assert s.tts.said == [SAY_AGAIN]  # it asked again once, then gave up


def test_a_question_that_was_not_understood_is_asked_again_and_then_answered():
    s, _, _, asked = build(stt=FakeStt(script=[""], text="where are my keys"))
    s.talk()
    s.wait()
    assert s.tts.said[0] == SAY_AGAIN and asked == ["where are my keys"] and s.problem == ""


def _boom(_):
    raise RuntimeError("model down")


@pytest.mark.parametrize("fault", [{"stt": FakeStt(fail=True)}, {"tts": FakeTts(fail=True)}, {"ask": _boom}])
def test_a_failing_stage_is_reported_and_the_next_question_still_works(fault):
    s, *_ = build(**fault)
    s.talk()
    s.wait()
    assert s.state == "idle" and s.problem
    s.mic.chunks = s.mic.chunks  # same fake room again
    s.stt, s.tts, s.ask = FakeStt(), FakeTts(), lambda t: ("Fine.", 0)
    s._quiet_until = 0
    s.talk()
    s.wait()
    assert s.problem == "" and s.answer == "Fine."


def test_off_means_off():
    s, mic, *_ = build()
    assert s.set_enabled(False)["state"] == "off"
    assert s.talk()["state"] == "off" and mic.opened == 0
    assert s.set_enabled(True)["state"] == "idle"


def test_stop_drops_what_was_heard():
    mic = FakeMic(silence(12) + tone(20), endless=True)
    s, _, speaker, asked = build(mic=mic, end_silence_s=60.0, max_seconds=600.0, wait_for_speech_s=60.0)
    s.talk()
    wait_for(s, "listening")
    s.stop()
    s.wait()
    assert s.state == "idle" and not s.stt.heard and not speaker.played and not asked


def test_the_recording_is_not_kept():
    s, *_ = build()
    s.talk()
    s.wait()
    assert not [k for k, v in vars(s).items() if isinstance(v, (bytes, bytearray)) and k != "_chime"]  # _chime is a generated sound, not audio from the room


# -- "Hey Atlas" --------------------------------------------------------------------------------------------------------

WAKE = dict(wake_word=True, wake_span_s=0.6, wake_step_s=0.3)  # a 20-chunk window every 10 chunks, so a short fake clip has several


def test_the_wake_listener_is_off_unless_asked_for():
    s, mic, *_ = build()
    s.start_wake()
    time.sleep(0.1)
    assert mic.opened == 0 and s.status()["wake"] is False


def test_hey_atlas_answers_whats_up_then_takes_the_question():
    mic = FakeMic(opens=[question(), question(), silence(5)])  # the wake phrase, then the question, then a quiet room
    s, mic, speaker, asked = build(mic=mic, stt=FakeStt(script=["Hey Atlas.", "what time is it"]), **WAKE)
    assert s.status()["wake"] is True
    s.start_wake()
    until(lambda: asked == ["what time is it"] and len(s.tts.said) == 2 and s.state == "idle")
    assert s.tts.said[0] == "What's up?"
    assert s.heard == "what time is it" and s.problem == ""
    assert not speaker.mic_was_open_while_playing
    assert s.stt.quick[:2] == [True, False]  # the wake check uses the quick model, the question the careful one


def test_hey_atlas_followed_by_a_question_still_greets_first_and_asks_only_the_question():
    mic = FakeMic(opens=[question(), question(), silence(5)])
    s, _, _, asked = build(mic=mic, stt=FakeStt(script=["Hey Atlas, hey Atlas, where are my keys?", "hey Atlas where are my keys"]), **WAKE)
    s.start_wake()
    until(lambda: asked == ["where are my keys"] and s.state == "idle")  # "Hey Atlas" is not part of the question
    assert s.tts.said[0] == "What's up?"


def test_the_wake_phrase_is_heard_through_constant_noise():
    mic = FakeMic(tone(30, 3000), endless=True)  # a steady 0.09 level that never pauses: nothing for a loudness detector to find
    s, _, speaker, _ = build(mic=mic, stt=FakeStt(script=["[MUSIC]", "Hey Atlas."]), **WAKE)
    s.start_wake()
    until(lambda: speaker.played)  # the second window was the wake phrase: "What's up?" is spoken
    assert s.tts.said[0] == "What's up?"


def test_speech_that_is_not_the_wake_phrase_is_dropped():
    mic = FakeMic(opens=[question(), silence(5)])
    s, _, speaker, asked = build(mic=mic, stt=FakeStt(script=["I think we should order pizza"]), **WAKE)
    s.start_wake()
    until(lambda: s.stt.heard)  # it was transcribed to check for the wake phrase...
    time.sleep(0.2)
    assert asked == [] and speaker.played == [] and s.heard == "" and s.state == "idle"  # ...and that is all that happened


def test_a_tap_takes_the_microphone_from_the_wake_listener():
    mic = FakeMic(silence(20), endless=True, opens=[silence(20), question()])
    s, mic, _, asked = build(mic=mic, **WAKE)
    s.start_wake()
    until(lambda: mic.is_open)  # the wake listener has it
    s.talk()
    until(lambda: asked == ["where did I leave my keys"])
    assert s.problem == ""


def test_no_microphone_for_the_wake_listener_is_a_message_and_it_keeps_trying(monkeypatch):
    monkeypatch.setattr("contexts.voice.session.WAKE_RETRY_S", 0.05)
    mic = FakeMic(missing=True)
    s, *_ = build(mic=mic, **WAKE)
    s.start_wake()
    until(lambda: s.problem.startswith("Not listening for Hey Atlas"))
    assert s.state == "idle"


def test_turning_the_voice_off_stops_the_wake_listener():
    mic = FakeMic(silence(20), endless=True)
    s, mic, *_ = build(mic=mic, **WAKE)
    s.start_wake()
    until(lambda: mic.is_open)
    s.set_enabled(False)
    until(lambda: not mic.is_open)


def test_the_wake_reply_is_made_once_and_reused():
    mic = FakeMic(opens=[question(), question(), silence(5), question(), question(), silence(5)])
    s, mic, speaker, asked = build(mic=mic, stt=FakeStt(script=["Hey Atlas.", "first one", "Hey Atlas.", "second one"]), **WAKE)
    s.start_wake()
    until(lambda: len(asked) == 2 and s.state == "idle")
    assert s.tts.said.count("What's up?") == 1  # Piper was asked for it once, the second wake reused it
    assert [p for p, _ in speaker.played].count(speaker.played[0][0]) >= 2


# -- the first sentence is spoken while the rest of the answer is still coming ----------------------------------------------

def streaming(pieces, answer, citations=0, gate=None):
    """An ask_stream that hands over `pieces` one by one (waiting on `gate` before the last one), then the whole answer."""
    def ask_stream(text, on_piece):
        for i, piece in enumerate(pieces):
            if gate is not None and i == len(pieces) - 1:
                gate.wait(3)
            on_piece(piece)
        return answer, citations
    return ask_stream


def build_streaming(ask_stream, **kw):
    s, mic, speaker, asked = build(**kw)
    s.ask_stream = ask_stream
    return s, mic, speaker, asked


def test_the_first_sentence_is_spoken_before_the_rest_of_the_answer_exists():
    gate = threading.Event()
    pieces = ["Your keys are on the desk.", " I saw them there at 8 PM.", " The laptop is next to them."]
    s, mic, speaker, _ = build_streaming(streaming(pieces, "".join(pieces), 2, gate))
    s.talk()
    until(lambda: len(speaker.played) == 1)  # the first sentence is already out loud while the last piece is withheld
    assert s.tts.said == ["Your keys are on the desk.", "I saw them there at 8 PM."][: len(s.tts.said)] and s.state == "speaking"
    gate.set()
    s.wait()
    assert s.tts.said == ["Your keys are on the desk.", "I saw them there at 8 PM.", "The laptop is next to them.",
                          "Based on 2 pieces of evidence."]
    assert s.answer == "".join(pieces) and s.problem == "" and len(speaker.played) == 4 and not speaker.mic_was_open_while_playing


def test_an_answer_that_was_not_streamed_is_spoken_whole():
    s, *_ = build_streaming(streaming([], "Nothing on the desk.", 1))
    s.talk()
    s.wait()
    assert s.tts.said == ["Nothing on the desk. Based on 1 piece of evidence."]


def test_a_tap_while_a_streamed_answer_is_spoken_stops_the_rest():
    gate = threading.Event()
    pieces = ["First sentence is right here.", " Second sentence follows it.", " Third one comes last."]
    mic = FakeMic()
    speaker = FakeSpeaker(mic)
    speaker.block = threading.Event()  # the first sentence "plays" until released
    s, *_ = build_streaming(streaming(pieces, "".join(pieces), 0, gate), mic=mic, speaker=speaker)
    s.talk()
    wait_for(s, "speaking")
    s.talk()  # stop
    gate.set()
    s.wait()
    assert s.state == "idle" and len(speaker.played) == 1


def test_a_streamed_answer_that_fails_halfway_is_reported():
    def broken(text, on_piece):
        on_piece("Your keys are on the desk.")
        on_piece(" I think")
        raise RuntimeError("model down")

    s, *_ = build_streaming(broken)
    s.talk()
    s.wait()
    assert s.state == "idle" and "did not answer" in s.problem and s.tts.said == ["Your keys are on the desk."]


def test_no_speaker_leaves_a_streamed_answer_on_screen():
    mic = FakeMic()
    s, *_ = build_streaming(streaming(["Your keys are on the desk."], "Your keys are on the desk."), mic=mic,
                            speaker=FakeSpeaker(mic, missing=True))
    s.talk()
    s.wait()
    assert s.answer == "Your keys are on the desk." and "on screen" in s.problem


# -- the blip that says "I heard you" ---------------------------------------------------------------------------------------

def test_a_blip_plays_once_the_question_is_heard_and_before_the_answer():
    s, mic, speaker, asked = build(chime=True)
    s.talk()
    s.wait()
    assert asked == ["where did I leave my keys"]
    assert len(speaker.played) == 2 and speaker.played[0][0] == s._chime  # the blip, then the answer
    assert not speaker.mic_was_open_while_playing


def test_no_blip_when_nothing_was_heard():
    s, _, speaker, _ = build(mic=FakeMic(silence(100)), chime=True)
    s.talk()
    s.wait()
    assert speaker.played == []


def test_a_speaker_that_cannot_blip_does_not_stop_the_answer_from_being_found():
    mic = FakeMic()
    s, _, _, asked = build(mic=mic, speaker=FakeSpeaker(mic, missing=True), chime=True)
    s.talk()
    s.wait()
    assert asked == ["where did I leave my keys"] and s.answer


def test_the_blip_is_a_short_quiet_sound():
    from contexts.voice.audio import CHIME_RATE, chime, rms
    pcm = chime()
    assert 0.2 < len(pcm) / 2 / CHIME_RATE < 0.5 and 0.01 < rms(pcm) < 0.2
