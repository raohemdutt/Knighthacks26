import json
import subprocess
import wave
from pathlib import Path

import pytest

from contexts.voice.config import VoiceConfig
from contexts.voice.engines import PiperTts, SpeechEngineError, WhisperCli


def cfg(tmp_path) -> VoiceConfig:
    for name in ("whisper-cli", "ggml.bin", "piper", "voice.onnx"):
        (tmp_path / name).write_bytes(b"x")
    (tmp_path / "voice.onnx.json").write_text(json.dumps({"audio": {"sample_rate": 16000}}))
    return VoiceConfig(whisper_bin=str(tmp_path / "whisper-cli"), whisper_model=str(tmp_path / "ggml.bin"),
                       piper_bin=str(tmp_path / "piper"), piper_model=str(tmp_path / "voice.onnx"))


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
