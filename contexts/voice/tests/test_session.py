import threading
import time

import pytest

from contexts.voice.session import VoiceSession
from contexts.voice.tests.fakes import FakeMic, FakeSpeaker, FakeStt, FakeTts, make_config, silence, tone


def build(mic=None, speaker=None, stt=None, tts=None, ask=None, **cfg):
    mic = mic or FakeMic()
    speaker = speaker or FakeSpeaker(mic)
    asked = []

    def default_ask(text):
        asked.append(text)
        return "You left them on the desk at 15:40. See frames:cam-desk:1760000000000.", 2

    s = VoiceSession(make_config(**cfg), mic, speaker, stt or FakeStt(), tts or FakeTts(), ask or default_ask)
    return s, mic, speaker, asked


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
    assert not [v for v in vars(s).values() if isinstance(v, (bytes, bytearray))]
