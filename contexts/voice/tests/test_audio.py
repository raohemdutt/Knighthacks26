from contexts.voice.audio import record_utterance, rms
from contexts.voice.tests.fakes import question, silence, tone

KW = dict(rate=16000, speech_level=0.02, end_silence_s=0.6, wait_for_speech_s=1.0, max_seconds=3.0, should_stop=lambda: False)
CHUNK = len(tone(1)[0])


def test_rms_of_silence_and_a_loud_tone():
    assert rms(b"") == 0.0
    assert rms(silence(1)[0]) == 0.0
    assert 0.2 < rms(tone(1)[0]) < 0.3


def test_a_spoken_phrase_is_cut_out_with_its_lead_in():
    pcm = record_utterance(iter(question()), **KW)
    assert pcm is not None
    # 20 chunks of speech, the pause that ended it (0.6 s = 20 chunks), and a lead-in of at most 300 ms (10 chunks)
    assert 20 * CHUNK <= len(pcm) <= (20 + 20 + 10) * CHUNK


def test_a_quiet_room_returns_none():
    assert record_utterance(iter(silence(100)), **KW) is None


def test_a_single_click_is_not_speech():
    assert record_utterance(iter(silence(10) + tone(1) + silence(60)), **KW) is None


def test_a_steady_hiss_is_the_baseline_not_speech():
    assert record_utterance(iter(tone(40, 3000)), **KW) is None  # rms about 0.09, but it was there from the start


def test_stop_ends_the_recording_early():
    calls = {"n": 0}

    def stop():
        calls["n"] += 1
        return calls["n"] > 25

    pcm = record_utterance(iter(silence(20) + tone(200)), **{**KW, "should_stop": stop})
    assert pcm is not None and len(pcm) < 30 * CHUNK


def test_a_long_ramble_is_capped():
    pcm = record_utterance(iter(silence(20) + tone(1000)), **{**KW, "max_seconds": 1.0})
    assert pcm is not None and len(pcm) <= 40 * CHUNK
