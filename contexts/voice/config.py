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
    whisper_model: str = ""
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
    sample_rate: int = 16000  # what whisper wants; not a config key

    def __post_init__(self) -> None:
        home = Path(self.home).expanduser()
        self.whisper_bin = self.whisper_bin or str(home / "src/whisper.cpp-1.7.6/build/bin/whisper-cli")
        self.whisper_model = self.whisper_model or str(home / "models/ggml-tiny.en-q5_1.bin")
        self.piper_bin = self.piper_bin or str(home / "piper/piper")
        self.piper_model = self.piper_model or str(home / "models/en_US-lessac-low.onnx")


def load_config(path: str | Path = "config/voice.yaml") -> VoiceConfig:
    p = Path(path)
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) if p.is_file() else {}
    known = {f.name for f in fields(VoiceConfig)} - {"sample_rate"}
    return VoiceConfig(**{k: v for k, v in (raw or {}).items() if k in known})


def gateway_token() -> str | None:
    return os.environ.get("ATLAS_TOKEN") or None
