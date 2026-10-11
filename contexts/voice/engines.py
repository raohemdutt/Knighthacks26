"""Speech to text (whisper.cpp) and text to speech (Piper), each a short-lived process per use.

Neither model stays loaded between questions: on the 8 GB Jetson, whisper `tiny.en` peaks near 200 MB and Piper near
150 MB while they run, and 0 MB the rest of the time, so the LLM and the detector keep their memory (X16, step 0).
The `run` argument is subprocess.run; tests pass a fake."""

import io
import json
import os
import subprocess
import tempfile
import wave
from pathlib import Path

from contexts.voice.config import VoiceConfig
from contexts.voice.text import clean_transcript


class SpeechEngineError(RuntimeError):
    """A speech tool is missing or failed; the message is for the operator screen."""


def _scratch_dir() -> str:
    # RAM-backed where there is one, so a question's audio never touches the SD card
    return "/dev/shm" if os.path.isdir("/dev/shm") else tempfile.gettempdir()


def _audio_ctx(pcm_bytes: int, rate: int) -> int:
    """whisper encodes a full 30 s window (1500 frames, 50 per second) however short the clip is; sizing it to the clip plus
    two seconds to spare takes a 4 s question from about 1.5 s to 0.4 s on the Jetson, at no visible cost in accuracy."""
    seconds = pcm_bytes / 2 / rate
    return min(1500, max(256, int(seconds * 50) + 100))


class WhisperCli:
    def __init__(self, cfg: VoiceConfig, run=subprocess.run):
        self.cfg, self.run = cfg, run

    def ready(self) -> tuple[bool, str]:
        missing = [p for p in (self.cfg.whisper_bin, self.cfg.whisper_model) if not Path(p).is_file()]
        return (False, f"speech-to-text files missing: {', '.join(missing)}") if missing else (True, "whisper.cpp tiny.en")

    def transcribe(self, pcm: bytes) -> str:
        """16-bit mono PCM at cfg.sample_rate -> the words, or "" when there were none."""
        wav = io.BytesIO()
        with wave.open(wav, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(self.cfg.sample_rate)
            w.writeframes(pcm)
        # whisper-cli reads a file, so the question sits in a temp file for the 2 seconds it takes, then is deleted
        fd, path = tempfile.mkstemp(suffix=".wav", prefix="atlas-voice-", dir=_scratch_dir())
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(wav.getvalue())
            try:
                done = self.run([self.cfg.whisper_bin, "-m", self.cfg.whisper_model, "-f", path, "-t", str(self.cfg.whisper_threads),
                                 "-nt", "-np", "-l", "en", "-bs", "1", "-bo", "1", "-ac", str(_audio_ctx(len(pcm), self.cfg.sample_rate))],
                                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=45)  # greedy decoding can end mid-character
            except (OSError, subprocess.TimeoutExpired) as e:
                raise SpeechEngineError(f"speech-to-text failed: {e}") from e
        finally:
            Path(path).unlink(missing_ok=True)
        if done.returncode:
            raise SpeechEngineError(f"speech-to-text failed: {(done.stderr or '').strip()[-120:] or done.returncode}")
        return clean_transcript(done.stdout)


class PiperTts:
    def __init__(self, cfg: VoiceConfig, run=subprocess.run):
        self.cfg, self.run = cfg, run

    def ready(self) -> tuple[bool, str]:
        missing = [p for p in (self.cfg.piper_bin, self.cfg.piper_model) if not Path(p).is_file()]
        return (False, f"text-to-speech files missing: {', '.join(missing)}") if missing else (True, "Piper " + Path(self.cfg.piper_model).stem)

    def _rate(self) -> int:
        try:
            return int(json.loads(Path(self.cfg.piper_model + ".json").read_text(encoding="utf-8"))["audio"]["sample_rate"])
        except (OSError, KeyError, ValueError):
            return 22050

    def synthesize(self, text: str) -> tuple[bytes, int]:
        """Text -> (raw 16-bit mono PCM, sample rate). Straight from Piper's stdout; no file."""
        env = dict(os.environ, LD_LIBRARY_PATH=str(Path(self.cfg.piper_bin).parent))  # its own onnxruntime and espeak libraries
        try:
            done = self.run([self.cfg.piper_bin, "--model", self.cfg.piper_model, "--output-raw", "-q"], input=text.encode("utf-8"),
                            capture_output=True, timeout=60, env=env)
        except (OSError, subprocess.TimeoutExpired) as e:
            raise SpeechEngineError(f"text-to-speech failed: {e}") from e
        if done.returncode or not done.stdout:
            raise SpeechEngineError(f"text-to-speech failed: {(done.stderr or b'').decode(errors='replace').strip()[-120:] or done.returncode}")
        return done.stdout, self._rate()
