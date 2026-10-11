"""Speech to text (whisper.cpp) and text to speech (Piper), each a short-lived process per use.

Neither model stays loaded between questions: on the 8 GB Jetson, whisper (`base.en` for a question, `tiny.en` for the
wake listener) peaks near 180 MB and Piper near 170 MB while they run, and 0 MB the rest of the time, so the LLM and the
detector keep their memory (X16, step 0).
The `run` argument is subprocess.run; tests pass a fake."""

import io
import json
import logging
import os
import subprocess
import tempfile
import wave
from pathlib import Path

from contexts.voice.config import VoiceConfig
from contexts.voice.text import clean_transcript

log = logging.getLogger("atlas.voice")


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


def _model_name(path: str) -> str:
    """ggml-base.en-q5_1.bin -> base.en"""
    return Path(path).stem.removeprefix("ggml-").split("-")[0]


def _confidence(scores_path: str) -> float | None:
    """The mean of whisper's probability for each word piece it wrote (not its timing and control tokens). None when the
    file is missing or unreadable: then nothing is held back."""
    try:
        data = json.loads(Path(scores_path).read_text(encoding="utf-8", errors="replace"))
        odds = [float(t["p"]) for seg in data.get("transcription", []) for t in seg.get("tokens", [])
                if not str(t.get("text", "")).startswith(("[_", "<|"))]
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None
    return sum(odds) / len(odds) if odds else None


class WhisperCli:
    def __init__(self, cfg: VoiceConfig, run=subprocess.run):
        self.cfg, self.run = cfg, run

    def ready(self) -> tuple[bool, str]:
        missing = [p for p in (self.cfg.whisper_bin, self.cfg.whisper_model, self.cfg.wake_model) if not Path(p).is_file()]
        if missing:
            return False, f"speech-to-text files missing: {', '.join(missing)}"
        return True, "whisper.cpp " + _model_name(self.cfg.whisper_model)

    def transcribe(self, pcm: bytes, quick: bool = False) -> str:
        """16-bit mono PCM at cfg.sample_rate -> the words, or "" when there were none or whisper was not sure of them.

        quick (the wake listener, every 1.5 s): the small model, greedy. Otherwise (a question): the bigger model, beam
        search and the vocabulary prompt, and words whisper itself is unsure of come back as "" so the box asks again
        instead of answering something it misheard ("what caused sure to my wife?")."""
        wav = io.BytesIO()
        with wave.open(wav, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(self.cfg.sample_rate)
            w.writeframes(pcm)
        # whisper-cli reads a file, so the question sits in a temp file for the 2 seconds it takes, then is deleted
        fd, path = tempfile.mkstemp(suffix=".wav", prefix="atlas-voice-", dir=_scratch_dir())
        scores = path + ".json"  # whisper's odds for each word piece (-ojf), next to the audio and deleted with it
        c = self.cfg
        # -nf: no retry at a higher temperature, which is where whisper's random words come from on a hard clip
        cmd = [c.whisper_bin, "-m", c.wake_model if quick else c.whisper_model, "-f", path, "-t", str(c.whisper_threads),
               "-nt", "-np", "-l", "en", "-nf", "-sns", "-ac", str(_audio_ctx(len(pcm), c.sample_rate))]
        if quick:
            cmd += ["-bs", "1", "-bo", "1"]
        else:
            cmd += ["-bs", str(max(1, c.whisper_beam)), "-ojf", "-of", path]
            if c.whisper_prompt.strip():
                cmd += ["--prompt", c.whisper_prompt.strip()]
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(wav.getvalue())
            try:
                done = self.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=45)  # greedy decoding can end mid-character
            except (OSError, subprocess.TimeoutExpired) as e:
                raise SpeechEngineError(f"speech-to-text failed: {e}") from e
            confidence = None if quick else _confidence(scores)
        finally:
            Path(path).unlink(missing_ok=True)
            Path(scores).unlink(missing_ok=True)
        if done.returncode:
            raise SpeechEngineError(f"speech-to-text failed: {(done.stderr or '').strip()[-120:] or done.returncode}")
        text = clean_transcript(done.stdout)
        if c.debug_speech and not quick:
            log.info("heard %r (confidence %s)", text, "?" if confidence is None else f"{confidence:.2f}")
        if text and confidence is not None and confidence < c.min_confidence:
            return ""
        return text


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
        # A deeper voice with no extra tool: Piper speaks faster by the pitch factor and the audio is played back slower by
        # the same factor, so the pitch drops and the pace comes back to normal (times voice_speed).
        pitch = min(1.0, max(0.7, self.cfg.voice_pitch))
        length = pitch / min(1.5, max(0.5, self.cfg.voice_speed))
        try:
            done = self.run([self.cfg.piper_bin, "--model", self.cfg.piper_model, "--output-raw", "-q", "--length_scale", f"{length:.3f}"],
                            input=text.encode("utf-8"), capture_output=True, timeout=60, env=env)
        except (OSError, subprocess.TimeoutExpired) as e:
            raise SpeechEngineError(f"text-to-speech failed: {e}") from e
        if done.returncode or not done.stdout:
            raise SpeechEngineError(f"text-to-speech failed: {(done.stderr or b'').decode(errors='replace').strip()[-120:] or done.returncode}")
        return done.stdout, round(self._rate() * pitch)
