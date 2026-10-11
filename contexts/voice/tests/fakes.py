"""Fake audio in and out, so the whole voice loop runs on a laptop with no microphone, speaker or speech models."""

import struct
import time
from contextlib import contextmanager

from contexts.voice.audio import AudioUnavailable
from contexts.voice.config import VoiceConfig
from contexts.voice.engines import SpeechEngineError

CHUNK_SAMPLES = 480  # 30 ms at 16 kHz


def tone(chunks: int, level: int = 8000) -> list[bytes]:
    return [struct.pack("<h", level) * CHUNK_SAMPLES for _ in range(chunks)]


def silence(chunks: int) -> list[bytes]:
    return tone(chunks, 0)


def question() -> list[bytes]:
    """Quiet room, a spoken question, then the pause that ends it."""
    return silence(12) + tone(20) + silence(45)


def make_config(**over) -> VoiceConfig:
    return VoiceConfig(**{"wait_for_speech_s": 1.0, "end_silence_s": 0.6, "max_seconds": 3.0, "chime": False, **over})


class FakeMic:
    def __init__(self, chunks: list[bytes] | None = None, missing: bool = False, endless: bool = False,
                 opens: list[list[bytes]] | None = None):
        self.chunks, self.missing, self.endless = chunks if chunks is not None else question(), missing, endless
        self.opens = opens  # what each successive open hears (the last one repeats); overrides `chunks`
        self._now = self.chunks
        self.is_open = False
        self.opened = 0

    def available(self):
        return (False, "no microphone (fake)") if self.missing else (True, "fake mic")

    @contextmanager
    def open(self):
        if self.missing:
            raise AudioUnavailable("no such device (fake)")
        self.is_open = True
        self.opened += 1
        self._now = self.opens[min(self.opened, len(self.opens)) - 1] if self.opens else self.chunks
        try:
            yield self._frames()
        finally:
            self.is_open = False

    def _frames(self):
        now = self._now
        yield from now
        while self.endless:  # someone who keeps talking until the button is tapped again
            time.sleep(0.005)
            yield now[-1] if now else b""


class FakeSpeaker:
    def __init__(self, mic: FakeMic | None = None, missing: bool = False):
        self.mic, self.missing = mic, missing
        self.played: list[tuple[bytes, int]] = []
        self.mic_was_open_while_playing = False
        self.block = None  # a threading.Event the test releases to end the "speech"

    def available(self):
        return (False, "speaker not connected (fake)") if self.missing else (True, "fake speaker")

    def ensure_connected(self):
        return not self.missing

    def play(self, pcm, rate, stop):
        if self.missing:
            raise AudioUnavailable("speaker not connected (fake)")
        if self.mic is not None and self.mic.is_open:
            self.mic_was_open_while_playing = True
        self.played.append((pcm, rate))
        if self.block is not None:
            while not stop.is_set() and not self.block.is_set():
                stop.wait(0.01)


class FakeStt:
    def __init__(self, text: str = "where did I leave my keys", fail: bool = False, script: list[str] | None = None):
        self.text, self.fail = text, fail
        self.script = list(script or [])  # what the first transcriptions say, in order; then `text`
        self.heard: list[int] = []  # byte length of each recording given to it
        self.quick: list[bool] = []  # whether each one was the wake listener's quick check

    def ready(self):
        return True, "fake whisper"

    def transcribe(self, pcm: bytes, quick: bool = False) -> str:
        self.heard.append(len(pcm))
        self.quick.append(quick)
        if self.fail:
            raise SpeechEngineError("speech-to-text failed (fake)")
        return self.script.pop(0) if self.script else self.text


class FakeTts:
    def __init__(self, fail: bool = False):
        self.fail = fail
        self.said: list[str] = []

    def ready(self):
        return True, "fake piper"

    def synthesize(self, text: str):
        if self.fail:
            raise SpeechEngineError("text-to-speech failed (fake)")
        self.said.append(text)
        return b"\x01\x00" * 2205, 22050
