"""Settings for the voice service: config/voice.yaml, with the gateway's token from the environment."""

import os
from dataclasses import dataclass, fields
from pathlib import Path

import yaml


@dataclass
class VoiceConfig:
    enabled: bool = True
    port: int = 8091
    ask_url: str = "http://127.0.0.1:8000"
    home: str = "~/atlas/voice"
    whisper_bin: str = ""
    whisper_model: str = ""  # for the question: base.en, which mishears far less than tiny.en
    wake_model: str = ""  # for "Hey Atlas", checked every 1.5 s: tiny.en, the fast one
    piper_bin: str = ""
    piper_model: str = ""
    mic_device: str = "plughw:CARD=Webcam,DEV=0"
    speaker_mac: str = ""
    speaker_sink: str = ""
    max_seconds: float = 8.0
    wait_for_speech_s: float = 5.0
    end_silence_s: float = 1.0
    speech_level: float = 0.03
    whisper_threads: int = 4
    whisper_beam: int = 5  # beam search for the question (1 = greedy); the wake listener is always greedy
    whisper_prompt: str = ""  # words whisper should expect in a question (names, places, course words)
    min_confidence: float = 0.6  # a question heard with less (0 to 1, whisper's own token odds) is asked again, not answered
    voice_pitch: float = 1.0  # below 1 = a deeper voice (0.88 is about two semitones down); speed is unchanged
    voice_speed: float = 1.0  # below 1 = slower, clearer speech
    wake_word: bool = False  # always listening for "Hey Atlas" (config/voice.yaml turns it on)
    wake_reply: str = "What's up?"
    wake_span_s: float = 3.0  # how much of the most recent sound is checked for the wake words...
    wake_step_s: float = 1.5  # ...this often (the span must be longer than the step plus "Hey Atlas", about a second)
    debug_wake: bool = False  # log what the wake listener heard, to tune it; off because it writes what the room said
    debug_speech: bool = False  # log each question heard (with whisper's confidence) and each sentence spoken
    chime: bool = True  # a short blip once the question has been heard, so you know it is working on it
    sample_rate: int = 16000  # what whisper wants; not a config key

    def __post_init__(self) -> None:
        home = Path(self.home).expanduser()
        self.whisper_bin = self.whisper_bin or str(home / "src/whisper.cpp-1.7.6/build/bin/whisper-cli")
        self.whisper_model = self.whisper_model or str(home / "models/ggml-base.en-q5_1.bin")
        self.wake_model = self.wake_model or str(home / "models/ggml-tiny.en-q5_1.bin")
        self.piper_bin = self.piper_bin or str(home / "piper/piper")
        self.piper_model = self.piper_model or str(home / "models/en_US-ryan-medium.onnx")


def load_config(path: str | Path = "config/voice.yaml") -> VoiceConfig:
    p = Path(path)
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) if p.is_file() else {}
    known = {f.name for f in fields(VoiceConfig)} - {"sample_rate"}
    return VoiceConfig(**{k: v for k, v in (raw or {}).items() if k in known})


def gateway_token() -> str | None:
    return os.environ.get("ATLAS_TOKEN") or None
