"""The box's ears and mouth: arecord for the webcam microphone, paplay (or aplay) for the speaker.

Both are small subprocesses, so a missing device is an AudioUnavailable error with a plain reason, never a crash.
Audio only ever moves between a pipe and these tools; nothing is written to disk."""

import array
import math
import shutil
import subprocess
import threading
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager

CHUNK_MS = 30
PRE_ROLL_MS = 300
NOISE_CALIBRATION_MS = 300
WARMUP_CHUNKS = 5


CHIME_RATE = 24000


def chime(rate: int = CHIME_RATE) -> bytes:
    """A short rising two-note blip, raw 16-bit mono: "I got your question, working on it". It starts with a little
    silence because a Bluetooth speaker that has gone idle swallows the first moments of a sound."""
    out = array.array("h", [0] * (rate * 120 // 1000))
    for freq, ms in ((660, 90), (880, 130)):
        n = rate * ms // 1000
        for i in range(n):
            envelope = min(1.0, i / (rate * 0.005)) * math.exp(-3.5 * i / n)  # a click-free start, then a soft fade
            out.append(int(9000 * envelope * math.sin(2 * math.pi * freq * i / rate)))
    return out.tobytes()


class AudioUnavailable(RuntimeError):
    """No microphone or no speaker right now; the message says why, in words for the operator screen."""


def rms(pcm: bytes) -> float:
    """Loudness of 16-bit little-endian samples, 0 to 1."""
    samples = array.array("h")
    samples.frombytes(pcm[: len(pcm) // 2 * 2])
    if not samples:
        return 0.0
    return math.sqrt(sum(s * s for s in samples) / len(samples)) / 32768.0


def record_utterance(frames: Iterator[bytes], *, rate: int, speech_level: float, end_silence_s: float,
                     wait_for_speech_s: float, max_seconds: float, should_stop: Callable[[], bool]) -> bytes | None:
    """Pull chunks until a spoken phrase has started and then gone quiet. None when nobody spoke or it was stopped.

    The room's own noise is measured on the first half second (after the microphone's warm-up); speech is louder than
    `speech_level` or three times that noise, whichever is higher, for three chunks in a row. Once speaking, anything
    above 60% of that bar still counts as speech, so a soft word does not end the question. A short pre-roll is kept so
    the first word is not clipped."""
    chunk_s = CHUNK_MS / 1000
    pre_roll: deque[bytes] = deque(maxlen=PRE_ROLL_MS // CHUNK_MS)
    calibration: list[float] = []
    kept: list[bytes] = []
    loud_run = 0
    quiet_s = 0.0
    elapsed = 0.0
    threshold = speech_level
    speaking = False
    for n, chunk in enumerate(frames):
        if should_stop():
            break
        elapsed += chunk_s
        if n < WARMUP_CHUNKS:  # the first chunks off a USB microphone are often zeros
            continue
        level = rms(chunk)
        if len(calibration) < NOISE_CALIBRATION_MS // CHUNK_MS:
            calibration.append(level)
            threshold = max(speech_level, 3 * sorted(calibration)[len(calibration) // 4])
        if not speaking:
            pre_roll.append(chunk)
            loud_run = loud_run + 1 if level > threshold else 0
            if loud_run >= 3:
                speaking = True
                kept = list(pre_roll)
            elif elapsed >= wait_for_speech_s:
                return None
            continue
        kept.append(chunk)
        quiet_s = 0.0 if level > threshold * 0.6 else quiet_s + chunk_s
        if quiet_s >= end_silence_s or elapsed >= max_seconds:
            break
    return b"".join(kept) if speaking else None


class Microphone:
    """arecord, raw 16-bit mono, read in 30 ms chunks."""

    def __init__(self, device: str, rate: int):
        self.device, self.rate = device, rate

    def available(self) -> tuple[bool, str]:
        if not shutil.which("arecord"):
            return False, "arecord is not installed"
        try:
            names = subprocess.run(["arecord", "-L"], capture_output=True, text=True, timeout=5).stdout
        except (OSError, subprocess.TimeoutExpired):
            return False, "could not list the microphones"
        if self.device not in names.splitlines():
            return False, f"no microphone {self.device} (is the webcam plugged in?)"
        return True, self.device

    @contextmanager
    def open(self) -> Iterator[Iterator[bytes]]:
        if not shutil.which("arecord"):
            raise AudioUnavailable("arecord is not installed")
        proc = subprocess.Popen(["arecord", "-q", "-D", self.device, "-f", "S16_LE", "-r", str(self.rate), "-c", "1", "-t", "raw"],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        size = self.rate * CHUNK_MS // 1000 * 2
        ended = False

        def chunks() -> Iterator[bytes]:
            nonlocal ended
            while True:
                data = proc.stdout.read(size)
                if len(data) < size:
                    ended = True
                    return
                yield data

        try:
            yield chunks()
            if ended:  # arecord stopped by itself: no such device, or it is busy
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    pass
                why = proc.stderr.read().decode(errors="replace").strip()[:120] if proc.poll() is not None else ""
                raise AudioUnavailable(f"microphone failed: {why or 'it stopped recording'}")
        finally:
            proc.kill()
            proc.wait()
            for pipe in (proc.stdout, proc.stderr):
                pipe.close()


class Speaker:
    """paplay into PulseAudio, so a paired Bluetooth speaker is just the default output. With `mac` set, the speaker
    is reconnected first if it dropped (it needs to have been paired and trusted once; see jetson/README.md)."""

    def __init__(self, mac: str = "", sink: str = ""):
        self.mac, self.sink = mac, sink

    def _bluetooth_connected(self) -> bool:
        try:
            info = subprocess.run(["bluetoothctl", "info", self.mac], capture_output=True, text=True, timeout=5).stdout
        except (OSError, subprocess.TimeoutExpired):
            return False
        return "Connected: yes" in info

    def ensure_connected(self) -> bool:
        if not self.mac or self._bluetooth_connected():
            return True
        try:
            subprocess.run(["bluetoothctl", "connect", self.mac], capture_output=True, timeout=15)
        except (OSError, subprocess.TimeoutExpired):
            return False
        return self._bluetooth_connected()

    def available(self) -> tuple[bool, str]:
        if not (shutil.which("paplay") or shutil.which("aplay")):
            return False, "no paplay or aplay installed"
        if self.mac:
            if not shutil.which("bluetoothctl"):
                return False, "bluetoothctl is not installed"
            if not self._bluetooth_connected():
                return False, f"Bluetooth speaker {self.mac} is not connected"
            return True, f"Bluetooth {self.mac}"
        return True, "default output"

    def play(self, pcm: bytes, rate: int, stop: threading.Event) -> None:
        """Raw 16-bit mono. Returns when finished or when `stop` is set."""
        if not self.ensure_connected():
            raise AudioUnavailable(f"Bluetooth speaker {self.mac} is not connected")
        if shutil.which("paplay"):
            cmd = ["paplay", "--raw", f"--rate={rate}", "--format=s16le", "--channels=1"]
            if self.sink:
                cmd.append(f"--device={self.sink}")
        elif shutil.which("aplay"):
            cmd = ["aplay", "-q", "-t", "raw", "-f", "S16_LE", "-r", str(rate), "-c", "1"]
        else:
            raise AudioUnavailable("no paplay or aplay installed")
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            step = rate * 2 // 4  # a quarter second, so a stop is heard quickly
            for i in range(0, len(pcm), step):
                if stop.is_set():
                    return
                proc.stdin.write(pcm[i : i + step])
            proc.stdin.close()
            while proc.poll() is None:
                if stop.wait(0.1):
                    return
            if proc.returncode:
                raise AudioUnavailable(f"speaker failed: {proc.stderr.read().decode(errors='replace').strip()[:120] or proc.returncode}")
        except (BrokenPipeError, OSError) as e:
            raise AudioUnavailable(f"speaker failed: {e}") from e
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.wait()
            proc.stderr.close()
