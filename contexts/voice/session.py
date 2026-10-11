"""The voice loop as a small state machine: off, idle, listening, thinking, speaking.

One thing at a time. A tap starts listening, and so does saying "Hey Atlas" when `wake_word` is on (see _wake_loop); a tap
while the box is speaking just stops the speech, so the microphone is never open while the speaker is playing. Every
failure (no mic, no speaker, no answer) is written to `problem` and the loop returns to idle; nothing here raises to the
caller.

Audio goes mic -> memory -> whisper -> text. Only the text goes on (to the gateway's ask path); the audio is dropped. While
it waits for "Hey Atlas" the box transcribes what it hears on the box itself and throws away everything that is not the
wake phrase: nothing is kept, logged, or sent anywhere."""

import contextlib
import logging
import queue
import threading
import time
from collections import deque
from collections.abc import Callable

from contexts.voice.audio import CHIME_RATE, CHUNK_MS, AudioUnavailable, chime, record_utterance, rms
from contexts.voice.config import VoiceConfig
from contexts.voice.engines import SpeechEngineError
from contexts.voice.text import (MAX_SPOKEN_CHARS, SentenceStream, clean_for_speech, evidence_line, find_wake, speech_for_answer,
                                 without_wake)

log = logging.getLogger("atlas.voice")

PROBE_EVERY_S = 10.0
MIN_WAKE_LEVEL = 0.003  # a window quieter than this is silence and is not transcribed
WAKE_RETRY_S = 5.0  # after the wake listener hits a missing microphone or speech tool
AFTER_SPEECH_QUIET_S = 0.5  # the speaker's tail must not be heard as the next question
SAY_AGAIN = "Sorry, I didn't catch that. Say it again?"  # someone spoke, but not clearly enough to answer

# ask(text) -> (answer, citation count): POST /api/ask on the gateway
Ask = Callable[[str], tuple[str, int]]
# ask_stream(text, on_piece) -> (answer, citation count): the same, handing over each piece of the answer as it is written
AskStream = Callable[[str, Callable[[str], None]], tuple[str, int]]


class VoiceSession:
    def __init__(self, cfg: VoiceConfig, mic, speaker, stt, tts, ask: Ask, clock: Callable[[], float] = time.monotonic,
                 ask_stream: AskStream | None = None):
        self.cfg, self.mic, self.speaker, self.stt, self.tts, self.ask, self.clock = cfg, mic, speaker, stt, tts, ask, clock
        self.ask_stream = ask_stream  # when given, the first sentence is spoken while the rest of the answer is still coming
        self._lock = threading.Lock()
        self._state = "idle"
        self._enabled = cfg.enabled
        self._finish = threading.Event()  # tap again: that is the whole question
        self._cancel = threading.Event()  # stop everything, drop what was heard
        self._stop_speech = threading.Event()
        self._thread: threading.Thread | None = None
        self._mic_lock = threading.Lock()  # one reader of the microphone at a time: a turn or the wake listener
        self._closing = threading.Event()
        self._wake_thread: threading.Thread | None = None
        self._chime = chime()
        self._greeting: tuple[bytes, int] | None = None  # the wake reply, spoken once and kept: it never changes
        self._quiet_until = 0.0
        self._ask_after = True  # False: only listen and write down what was said (the Ask screen asks and speaks itself)
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
                "seconds_in_state": round(self.clock() - self.since, 1), "wake": self.cfg.wake_word and self._enabled,
                **self._devices()}

    # -- the controls -----------------------------------------------------------------------------------------------

    def talk(self, ask: bool = True) -> dict:
        """The push-to-talk button. Idle: start listening. Listening: that was the whole question. Speaking: stop talking.
        Thinking: ignored (the answer is on its way). With ask=False the box only listens and keeps what it heard in
        `heard`: the Ask screen on the phone or laptop asks and shows the answer itself, so it gets the box's microphone
        and nothing else."""
        with self._lock:
            if not self._enabled:
                return self.status()
            if self._state == "listening":
                self._finish.set()
            elif self._state == "speaking":
                self._stop_speech.set()
            elif self._state == "idle" and self.clock() >= self._quiet_until:
                self.problem, self.heard, self.answer = "", "", ""
                self._ask_after = ask
                self._finish.clear()
                self._cancel.clear()
                self._stop_speech.clear()
                self._set("listening")
                started = self.status()  # taken before the turn runs, so the caller sees "listening" even if it ends at once
                self._thread = threading.Thread(target=self._run, name="voice-turn", daemon=True)
                self._thread.start()
                return started
        return self.status()

    def say(self, text: str) -> dict:
        """Speak this text on the box's speaker: the phone's Ask screen sends the answer here, so it comes out of the
        speaker on the box and not the phone. Only while idle (never over a question being heard or an answer being
        spoken); a failure goes to `problem` like any other turn."""
        said = speech_for_answer(text)
        with self._lock:
            if not self._enabled or self._state != "idle":
                return self.status()
            self.problem = ""
            self._stop_speech.clear()
            self._set("thinking")
            started = self.status()
            self._thread = threading.Thread(target=self._speak, args=(said,), name="voice-say", daemon=True)
            self._thread.start()
            return started

    def _speak(self, said: str) -> None:
        try:
            pcm, rate = self.tts.synthesize(said)
            self._set("speaking")
            self.speaker.play(pcm, rate, self._stop_speech)
        except (SpeechEngineError, AudioUnavailable) as e:
            self.problem = f"Couldn't speak ({e})."
        except Exception as e:  # nothing may take the service down
            log.exception("voice say failed")
            self.problem = f"voice error: {e}"
        finally:
            self._quiet_until = self.clock() + AFTER_SPEECH_QUIET_S
            self._set("idle")

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

    # -- "Hey Atlas": always listening, only for the wake phrase ------------------------------------------------------

    def start_wake(self) -> None:
        """Start listening for "Hey Atlas" in the background (a no-op unless `wake_word` is on)."""
        if self.cfg.wake_word and not (self._wake_thread and self._wake_thread.is_alive()):
            self._wake_thread = threading.Thread(target=self._wake_loop, name="voice-wake", daemon=True)
            self._wake_thread.start()

    def close(self) -> None:
        self._closing.set()
        self.stop()
        if self._wake_thread:
            self._wake_thread.join(2.0)

    def _wake_ready(self) -> bool:
        return self._enabled and self._state == "idle" and self.clock() >= self._quiet_until and not self._closing.is_set()

    def _wake_loop(self) -> None:
        """Whenever the box is idle, listen for "Hey Atlas" and start a turn when it is said. The microphone yields at
        once to a tap, a spoken answer, or a turn."""
        try:
            self._greeting = self.tts.synthesize(self.cfg.wake_reply)  # so the first "What's up?" does not wait for Piper
        except Exception:
            log.warning("could not prepare the wake reply; it will be made when needed")
        while not self._closing.is_set():
            if not self._wake_ready():
                self._closing.wait(0.2)
                continue
            try:
                if self._listen_for_wake():
                    self._start_wake_turn()
                else:
                    self._closing.wait(0.05)  # a turn took the microphone
            except (AudioUnavailable, SpeechEngineError) as e:
                if self._wake_ready():
                    self.problem = f"Not listening for Hey Atlas ({e})."
                self._closing.wait(WAKE_RETRY_S)
            except Exception:  # nothing may take the service down
                log.exception("wake listener failed")
                self._closing.wait(WAKE_RETRY_S)

    def _listen_for_wake(self) -> bool:
        """Keep the microphone open and, every `wake_step_s`, transcribe the last `wake_span_s` of sound. True once someone
        said "Hey Atlas". It does not wait for a pause in the sound: music, a fan or a crowd in the room would never
        leave one, and nothing is missed while a window is being transcribed, because the microphone stays open.
        Anything that is not the wake phrase is dropped unread."""
        c = self.cfg
        span, step = round(c.wake_span_s * 1000 / CHUNK_MS), round(c.wake_step_s * 1000 / CHUNK_MS)
        ring: deque[bytes] = deque(maxlen=span)
        waiting: queue.Queue[bytes | None] = queue.Queue(maxsize=1)  # the newest window not yet transcribed
        found = threading.Event()
        failure: list[Exception] = []

        def transcribe_windows() -> None:
            try:
                while (pcm := waiting.get()) is not None:
                    heard = self.stt.transcribe(pcm, quick=True)
                    del pcm
                    if c.debug_wake:
                        log.info("wake listener heard %r", heard)
                    if find_wake(heard) is not None:
                        found.set()
                        return
            except Exception as e:
                failure.append(e)

        worker = threading.Thread(target=transcribe_windows, name="voice-wake-stt", daemon=True)
        worker.start()
        try:
            with self._mic_lock:
                if not self._wake_ready():
                    return False
                with self.mic.open() as frames:
                    for n, chunk in enumerate(frames, 1):
                        if found.is_set() or failure or not self._wake_ready():
                            break
                        ring.append(chunk)
                        if n % step == 0:
                            window = b"".join(ring)
                            if rms(window) >= MIN_WAKE_LEVEL:  # dead silence has nothing to transcribe
                                with contextlib.suppress(queue.Empty):
                                    waiting.get_nowait()  # a newer window replaces one that has not been started
                                waiting.put_nowait(window)
        finally:
            while worker.is_alive():  # let the window in progress finish, then stop the worker
                try:
                    waiting.put(None, timeout=0.2)
                    break
                except queue.Full:
                    continue
            worker.join(5.0)
        if failure:
            raise failure[0]
        return found.is_set()

    def _start_wake_turn(self) -> None:
        with self._lock:
            if not self._wake_ready():  # a tap got there first, or the voice was turned off while it was transcribing
                return
            self.problem, self.heard, self.answer = "", "", ""
            self._ask_after = True
            self._finish.clear()
            self._cancel.clear()
            self._stop_speech.clear()
            self._set("thinking")
            self._thread = threading.Thread(target=self._run, kwargs={"greet": True}, name="voice-turn", daemon=True)
            self._thread.start()

    # -- one turn: listen, understand, ask, speak ---------------------------------------------------------------------

    def _run(self, greet: bool = False) -> None:
        """One turn. With `greet` (after "Hey Atlas") the wake reply is said first, then the box listens."""
        try:
            if greet and not self._greet():
                return
            text = self._listen_and_transcribe()
            if text == "" and self._ask_after and not self._cancel.is_set():
                # heard someone but not the words: ask once more rather than answer a guess
                if not self._greet(SAY_AGAIN):
                    return
                self._finish.clear()  # a tap that ended the first try must not end this one at once
                text = self._listen_and_transcribe()
                if text:
                    self.problem = ""
            if text and not self._cancel.is_set() and self._ask_after:
                self._ask_and_speak(text)
        except Exception as e:  # nothing may take the service down; say what happened instead
            log.exception("voice turn failed")
            self.problem = f"voice error: {e}"
        finally:
            self._quiet_until = self.clock() + AFTER_SPEECH_QUIET_S
            self._set("idle")

    def _start_chime(self) -> threading.Thread | None:
        """The question has been heard: blip while it is turned into words, so there is no silent wait. The microphone is
        already shut, and a speaker that cannot play it does not stop the turn."""
        if not self.cfg.chime:
            return None

        def play() -> None:
            try:
                self.speaker.play(self._chime, CHIME_RATE, self._stop_speech)
            except Exception:
                log.debug("chime not played", exc_info=True)

        t = threading.Thread(target=play, name="voice-chime", daemon=True)
        t.start()
        return t

    def _greet(self, line: str | None = None) -> bool:
        """Say the wake reply (or `line`), then get ready to listen. False when the turn should end instead."""
        try:
            if line is None:
                pcm, rate = self._greeting or self.tts.synthesize(self.cfg.wake_reply)
                self._greeting = pcm, rate
            else:
                pcm, rate = self.tts.synthesize(line)
            self._set("speaking")
            self.speaker.play(pcm, rate, self._stop_speech)
        except (SpeechEngineError, AudioUnavailable) as e:
            self.problem = f"Heard you but couldn't answer out loud ({e})."
            return False
        self._cancel.wait(AFTER_SPEECH_QUIET_S)  # the speaker's tail must not be heard as the question
        self._stop_speech.clear()  # a tap that cut the greeting short must not also cut the answer
        if self._cancel.is_set():
            return False
        self._set("listening")
        return True

    def _listen_and_transcribe(self) -> str | None:
        """The question in words. None when nobody spoke (or it was stopped); "" when someone spoke but whisper could not
        make out the words, or was not sure of them."""
        c = self.cfg
        try:
            with self._mic_lock, self.mic.open() as frames:
                pcm = record_utterance(frames, rate=c.sample_rate, speech_level=c.speech_level, end_silence_s=c.end_silence_s,
                                       wait_for_speech_s=c.wait_for_speech_s, max_seconds=c.max_seconds,
                                       should_stop=lambda: self._cancel.is_set() or self._finish.is_set())
        except AudioUnavailable as e:
            self.problem = f"No microphone: {e}"
            return None
        if self._cancel.is_set():
            return None
        if pcm is None:
            self.problem = "Didn't hear anything. Say Hey Atlas or tap, then speak." if c.wake_word else "Didn't hear anything. Tap and speak."
            return None
        self._set("thinking")
        blip = self._start_chime()
        try:
            text = self.stt.transcribe(pcm)
        except SpeechEngineError as e:
            self.problem = str(e)
            return None
        finally:
            del pcm  # the recording is gone as soon as it has been turned into words
            if blip:
                blip.join(3.0)  # done before anything else is said
        if c.wake_word:
            text = without_wake(text)  # "Hey Atlas" said again while it listens is not part of the question
        if not text:
            self.problem = "Didn't catch that. Tap and try again."
        self.heard = text
        return text

    def _ask_and_speak(self, text: str) -> None:
        if self.ask_stream is not None:
            return self._ask_and_speak_streaming(text)
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

    def _ask_and_speak_streaming(self, text: str) -> None:
        """Three steps run side by side: the answer arrives piece by piece, each finished sentence is turned into speech,
        and the speaker plays them in order. The first sentence is heard long before the last one is written."""
        sentences = SentenceStream()
        to_synth: queue.Queue[str | None] = queue.Queue()
        to_play: queue.Queue[tuple[bytes, int] | None] = queue.Queue()
        failed: dict[str, Exception] = {}
        final: dict[str, str] = {}
        queued = 0  # characters sent to be spoken, held to the same cap as a whole answer
        self.answer = ""

        def queue_sentence(raw: str) -> None:
            nonlocal queued
            said = clean_for_speech(raw)
            if said and (queued == 0 or queued + len(said) < MAX_SPOKEN_CHARS):
                if self.cfg.debug_speech:
                    log.info("saying %r", said)
                queued += len(said) + 1
                to_synth.put(said)

        def on_piece(piece: str) -> None:
            self.answer += piece
            for s in sentences.feed(piece):
                queue_sentence(s)

        def read_answer() -> None:
            try:
                answer, citations = self.ask_stream(text, on_piece)
                queue_sentence(sentences.flush())
                if queued == 0:  # nothing came piece by piece
                    to_synth.put(speech_for_answer(answer, citations))
                elif citations:
                    to_synth.put(evidence_line(citations))
                final["answer"] = answer
            except Exception as e:
                failed["ask"] = e
            finally:
                to_synth.put(None)

        def make_speech() -> None:
            try:
                while (said := to_synth.get()) is not None:
                    if self._cancel.is_set() or self._stop_speech.is_set():
                        continue
                    to_play.put(self.tts.synthesize(said))
            except SpeechEngineError as e:
                failed["speech"] = e
            finally:
                to_play.put(None)

        for work in (read_answer, make_speech):
            threading.Thread(target=work, name="voice-" + work.__name__, daemon=True).start()
        try:
            while (clip := to_play.get()) is not None and not self._cancel.is_set() and not self._stop_speech.is_set():
                if self._state != "speaking":
                    self._set("speaking")  # from here the microphone stays shut until the speaker is done
                self.speaker.play(clip[0], clip[1], self._stop_speech)
        except AudioUnavailable as e:
            failed["speech"] = e
        if "answer" in final:
            self.answer = final["answer"]
        if "ask" in failed:
            self.problem = f"The box did not answer: {failed['ask']}"
        elif "speech" in failed:
            self.problem = f"Couldn't speak the answer ({failed['speech']}). It is on screen."
