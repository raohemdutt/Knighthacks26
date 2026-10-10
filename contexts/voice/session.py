"""The voice loop as a small state machine: off, idle, listening, thinking, speaking.

One thing at a time. Only a tap starts listening, and a tap while the box is speaking just stops the speech, so the
microphone is never open while the speaker is playing. Every failure (no mic, no speaker, no answer) is written to
`problem` and the loop returns to idle; nothing here raises to the caller.

Audio goes mic -> memory -> whisper -> text. Only the text goes on (to the gateway's ask path); the audio is dropped."""

import logging
import threading
import time
from collections.abc import Callable

from contexts.voice.audio import AudioUnavailable, record_utterance
from contexts.voice.config import VoiceConfig
from contexts.voice.engines import SpeechEngineError
from contexts.voice.text import speech_for_answer

log = logging.getLogger("atlas.voice")

PROBE_EVERY_S = 10.0
AFTER_SPEECH_QUIET_S = 0.5  # the speaker's tail must not be heard as the next question

# ask(text) -> (answer, citation count): POST /api/ask on the gateway
Ask = Callable[[str], tuple[str, int]]


class VoiceSession:
    def __init__(self, cfg: VoiceConfig, mic, speaker, stt, tts, ask: Ask, clock: Callable[[], float] = time.monotonic):
        self.cfg, self.mic, self.speaker, self.stt, self.tts, self.ask, self.clock = cfg, mic, speaker, stt, tts, ask, clock
        self._lock = threading.Lock()
        self._state = "idle"
        self._enabled = cfg.enabled
        self._finish = threading.Event()  # tap again: that is the whole question
        self._cancel = threading.Event()  # stop everything, drop what was heard
        self._stop_speech = threading.Event()
        self._thread: threading.Thread | None = None
        self._quiet_until = 0.0
        self._probe: tuple[float, dict] | None = None
        self.problem = ""
        self.heard = ""
        self.answer = ""
        self.since = clock()

    # -- what the screen shows --------------------------------------------------------------------------------------

    @property
    def state(self) -> str:
        return self._state if self._enabled else "off"

    def _set(self, state: str) -> None:
        self._state = state
        self.since = self.clock()

    def _devices(self) -> dict:
        """Is there a mic, a speaker, and the speech files? Cached: the screen asks every second or so. A device that is
        in use right now is not probed (opening it again would fail); its last answer stands."""
        now = self.clock()
        if self._probe is None or now - self._probe[0] >= PROBE_EVERY_S:
            last = self._probe[1] if self._probe else {}
            info = {}
            for name, probe, busy in (("mic", self.mic.available, "listening"), ("speaker", self.speaker.available, "speaking"),
                                      ("speech_to_text", self.stt.ready, ""), ("text_to_speech", self.tts.ready, "")):
                if busy and self._state == busy and name in last:
                    info[name] = last[name]
                else:
                    ok, detail = probe()
                    info[name] = {"ok": ok, "detail": detail}
            self._probe = (now, info)
        return self._probe[1]

    def status(self) -> dict:
        return {"state": self.state, "running": True, "problem": self.problem, "heard": self.heard, "answer": self.answer,
                "seconds_in_state": round(self.clock() - self.since, 1), **self._devices()}

    # -- the controls -----------------------------------------------------------------------------------------------

    def talk(self) -> dict:
        """The push-to-talk button. Idle: start listening. Listening: that was the whole question. Speaking: stop talking.
        Thinking: ignored (the answer is on its way)."""
        with self._lock:
            if not self._enabled:
                return self.status()
            if self._state == "listening":
                self._finish.set()
            elif self._state == "speaking":
                self._stop_speech.set()
            elif self._state == "idle" and self.clock() >= self._quiet_until:
                self.problem, self.heard, self.answer = "", "", ""
                self._finish.clear()
                self._cancel.clear()
                self._stop_speech.clear()
                self._set("listening")
                started = self.status()  # taken before the turn runs, so the caller sees "listening" even if it ends at once
                self._thread = threading.Thread(target=self._run, name="voice-turn", daemon=True)
                self._thread.start()
                return started
        return self.status()

    def stop(self) -> dict:
        """Stop whatever is happening and drop what was heard."""
        self._cancel.set()
        self._stop_speech.set()
        self._finish.set()
        return self.status()

    def set_enabled(self, enabled: bool) -> dict:
        if not enabled:
            self.stop()
        self._enabled = enabled
        return self.status()

    def wait(self, timeout: float = 5.0) -> None:
        """For tests: block until the current turn is over."""
        t = self._thread
        if t:
            t.join(timeout)

    # -- one turn: listen, understand, ask, speak ---------------------------------------------------------------------

    def _run(self) -> None:
        try:
            text = self._listen_and_transcribe()
            if text and not self._cancel.is_set():
                self._ask_and_speak(text)
        except Exception as e:  # nothing may take the service down; say what happened instead
            log.exception("voice turn failed")
            self.problem = f"voice error: {e}"
        finally:
            self._quiet_until = self.clock() + AFTER_SPEECH_QUIET_S
            self._set("idle")

    def _listen_and_transcribe(self) -> str:
        c = self.cfg
        try:
            with self.mic.open() as frames:
                pcm = record_utterance(frames, rate=c.sample_rate, speech_level=c.speech_level, end_silence_s=c.end_silence_s,
                                       wait_for_speech_s=c.wait_for_speech_s, max_seconds=c.max_seconds,
                                       should_stop=lambda: self._cancel.is_set() or self._finish.is_set())
        except AudioUnavailable as e:
            self.problem = f"No microphone: {e}"
            return ""
        if self._cancel.is_set():
            return ""
        if pcm is None:
            self.problem = "Didn't hear anything. Tap and speak."
            return ""
        self._set("thinking")
        try:
            text = self.stt.transcribe(pcm)
        except SpeechEngineError as e:
            self.problem = str(e)
            return ""
        finally:
            del pcm  # the recording is gone as soon as it has been turned into words
        if not text:
            self.problem = "Didn't catch that. Tap and try again."
        self.heard = text
        return text

    def _ask_and_speak(self, text: str) -> None:
        try:
            answer, citations = self.ask(text)
        except Exception as e:
            self.problem = f"The box did not answer: {e}"
            return
        self.answer = answer
        if self._cancel.is_set():
            return
        said = speech_for_answer(answer, citations)
        try:
            pcm, rate = self.tts.synthesize(said)
            self._set("speaking")  # from here the microphone stays shut until the speaker is done
            self.speaker.play(pcm, rate, self._stop_speech)
        except (SpeechEngineError, AudioUnavailable) as e:
            self.problem = f"Couldn't speak the answer ({e}). It is on screen."
