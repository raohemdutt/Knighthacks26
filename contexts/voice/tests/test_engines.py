import json
import subprocess
import wave
from pathlib import Path

import pytest

from contexts.voice.config import VoiceConfig
from contexts.voice.engines import PiperTts, SpeechEngineError, WhisperCli


def cfg(tmp_path) -> VoiceConfig:
    for name in ("whisper-cli", "ggml.bin", "ggml-tiny.bin", "piper", "voice.onnx"):
        (tmp_path / name).write_bytes(b"x")
    (tmp_path / "voice.onnx.json").write_text(json.dumps({"audio": {"sample_rate": 16000}}))
    return VoiceConfig(whisper_bin=str(tmp_path / "whisper-cli"), whisper_model=str(tmp_path / "ggml.bin"),
                       wake_model=str(tmp_path / "ggml-tiny.bin"), piper_bin=str(tmp_path / "piper"),
                       piper_model=str(tmp_path / "voice.onnx"))


def test_whisper_gets_a_wav_in_a_temp_file_that_is_gone_afterwards(tmp_path):
    seen = {}

    def run(cmd, **kw):
        path = Path(cmd[cmd.index("-f") + 1])
        with wave.open(str(path)) as w:
            seen["format"] = (w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes())
        seen["path"] = path
        return subprocess.CompletedProcess(cmd, 0, stdout=" [BLANK_AUDIO]\n Where are my keys?\n", stderr="")

    text = WhisperCli(cfg(tmp_path), run=run).transcribe(b"\x00\x01" * 1600)
    assert text == "Where are my keys?"
    assert seen["format"] == (1, 2, 16000, 1600)
    assert not seen["path"].exists()  # no audio is kept


def test_whisper_failure_still_deletes_the_file_and_says_why(tmp_path):
    paths = []

    def run(cmd, **kw):
        paths.append(Path(cmd[cmd.index("-f") + 1]))
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="bad model")

    with pytest.raises(SpeechEngineError, match="bad model"):
        WhisperCli(cfg(tmp_path), run=run).transcribe(b"\x00\x00" * 160)
    assert not paths[0].exists()


def test_a_missing_binary_is_reported_not_raised_as_oserror(tmp_path):
    def run(cmd, **kw):
        raise FileNotFoundError("nope")

    with pytest.raises(SpeechEngineError):
        WhisperCli(cfg(tmp_path), run=run).transcribe(b"\x00\x00" * 160)


def test_ready_names_the_missing_files(tmp_path):
    ok, detail = WhisperCli(VoiceConfig(home=str(tmp_path))).ready()
    assert not ok and "missing" in detail
    ok, detail = PiperTts(VoiceConfig(home=str(tmp_path))).ready()
    assert not ok and "missing" in detail
    assert WhisperCli(cfg(tmp_path)).ready()[0] and PiperTts(cfg(tmp_path)).ready()[0]


def test_piper_returns_raw_audio_straight_from_stdout_with_the_voices_rate(tmp_path):
    calls = []

    def run(cmd, **kw):
        calls.append((cmd, kw))
        return subprocess.CompletedProcess(cmd, 0, stdout=b"\x01\x00" * 100, stderr=b"")

    pcm, rate = PiperTts(cfg(tmp_path), run=run).synthesize("Hello there.")
    cmd, kw = calls[0]
    assert (len(pcm), rate) == (200, 16000)
    assert "--output-raw" in cmd and "--output_file" not in cmd  # nothing is written to disk
    assert kw["input"] == b"Hello there." and "LD_LIBRARY_PATH" in kw["env"]


def test_piper_failure_is_an_engine_error(tmp_path):
    def run(cmd, **kw):
        return subprocess.CompletedProcess(cmd, 1, stdout=b"", stderr=b"no espeak data")

    with pytest.raises(SpeechEngineError, match="no espeak data"):
        PiperTts(cfg(tmp_path), run=run).synthesize("Hi.")


def test_whisper_encodes_only_as_much_audio_as_there_is(tmp_path):
    from contexts.voice.engines import _audio_ctx
    assert _audio_ctx(16000 * 2 * 4, 16000) == 300  # 4 s: 200 frames and two seconds to spare
    assert _audio_ctx(16000 * 2 * 1, 16000) == 256  # never smaller than this
    assert _audio_ctx(16000 * 2 * 40, 16000) == 1500  # never more than whisper's own 30 s window



def whisper_writing(text: str, odds: list[float], seen: dict):
    """A whisper-cli that prints `text` and, when asked for -ojf, writes those token odds where -of says."""
    def run(cmd, **kw):
        seen["cmd"] = cmd
        if "-ojf" in cmd:
            out = Path(cmd[cmd.index("-of") + 1] + ".json")
            tokens = [{"text": "[_BEG_]", "p": 0.01}] + [{"text": f" w{i}", "p": p} for i, p in enumerate(odds)]
            out.write_text(json.dumps({"transcription": [{"text": text, "tokens": tokens}]}))
            seen["scores"] = out
        return subprocess.CompletedProcess(cmd, 0, stdout=f" {text}\n", stderr="")
    return run


def test_a_question_uses_the_bigger_model_with_beam_search_and_no_random_retries(tmp_path):
    seen = {}
    c = cfg(tmp_path)
    c.whisper_prompt = "Atlas, Canvas, UCF."
    assert WhisperCli(c, run=whisper_writing("Where are my keys?", [0.9, 0.8], seen)).transcribe(b"\x00\x01" * 1600) == "Where are my keys?"
    cmd = seen["cmd"]
    assert cmd[cmd.index("-m") + 1] == c.whisper_model and cmd[cmd.index("-bs") + 1] == "5"
    assert "-nf" in cmd and cmd[cmd.index("--prompt") + 1] == "Atlas, Canvas, UCF."
    assert not seen["scores"].exists()  # the token odds are deleted with the audio


def test_a_question_whisper_is_unsure_of_comes_back_empty(tmp_path):
    seen = {}
    run = whisper_writing("What caused sure to my wife?", [0.3, 0.2, 0.5], seen)
    assert WhisperCli(cfg(tmp_path), run=run).transcribe(b"\x00\x01" * 1600) == ""
    assert not seen["scores"].exists()


def test_the_wake_check_uses_the_small_model_greedily(tmp_path):
    seen = {}
    c = cfg(tmp_path)
    assert WhisperCli(c, run=whisper_writing("Hey Atlas.", [], seen)).transcribe(b"\x00\x01" * 1600, quick=True) == "Hey Atlas."
    cmd = seen["cmd"]
    assert cmd[cmd.index("-m") + 1] == c.wake_model and cmd[cmd.index("-bs") + 1] == "1" and "-ojf" not in cmd


def test_unreadable_token_odds_hold_nothing_back(tmp_path):
    def run(cmd, **kw):
        Path(cmd[cmd.index("-of") + 1] + ".json").write_text("{not json")
        return subprocess.CompletedProcess(cmd, 0, stdout=" Where are my keys?\n", stderr="")

    assert WhisperCli(cfg(tmp_path), run=run).transcribe(b"\x00\x01" * 1600) == "Where are my keys?"


def test_a_deeper_voice_is_spoken_faster_and_played_slower(tmp_path):
    calls = []

    def run(cmd, **kw):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout=b"\x01\x00" * 100, stderr=b"")

    c = cfg(tmp_path)
    c.voice_pitch, c.voice_speed = 0.88, 0.95
    _, rate = PiperTts(c, run=run).synthesize("Hello there.")
    assert rate == round(16000 * 0.88)  # played back slower: the pitch drops
    assert calls[0][calls[0].index("--length_scale") + 1] == "0.926"  # spoken faster by as much, and a little slower overall
