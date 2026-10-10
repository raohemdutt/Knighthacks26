"""One perception model call at a time, paced in event time, and out of the way of people.

Model calls take seconds on the Jetson, and llama-server answers one request at a time. Captions and
desk object lists therefore share one queue:

- one call in flight at a time, whatever the camera or the kind;
- one queued request per (kind, camera): a newer one replaces it, so a slow model skips frames
  instead of building a backlog;
- at least `min_gap_s` of event time between the end of one call and the start of the next, so the
  model is free in between;
- no new call starts while a question is being answered, nor for `yield_after_question_s` after the
  answer. A question whose answer never comes stops counting after `yield_timeout_s`.

The clock is the newest `event.ts` the queue has been shown (`tick`), never the wall clock. Requests
start when an event arrives and the rules allow it: the cameras report every second, so a queued
request waits at most one sample longer than it has to.
"""

import asyncio
import logging
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any

from contracts.types import FrameRef

log = logging.getLogger("atlas.perception")

MIN_GAP_S = 2.0
YIELD_AFTER_QUESTION_S = 3.0
YIELD_TIMEOUT_S = 30.0  # a question unanswered for this long no longer holds perception back

Key = tuple[str, str]  # (kind, camera): "caption" or "objects"


@dataclass
class PerceptionStats:
    """What perception asked of the model, and what it cost. Counts since the service started."""

    captions_requested: int = 0  # every trigger: an enter or exit, or a periodic slot at the base cadence
    captions_sent: int = 0
    captions_skipped_redundant: int = 0  # periodic slots dropped because nothing had changed
    captions_coalesced: int = 0  # enters and exits folded into a caption just sent or already queued
    captions_without_frame: int = 0  # the trigger had no keyframe, or the keyframe was gone
    objects_requested: int = 0
    objects_sent: int = 0
    objects_coalesced: int = 0
    objects_without_frame: int = 0
    images_sent: int = 0
    bytes_before_scaling: int = 0  # what would have gone to the model: the keyframe, or the desk crop
    bytes_sent: int = 0

    def image(self, before: int, after: int) -> None:
        self.images_sent += 1
        self.bytes_before_scaling += before
        self.bytes_sent += after

    def as_dict(self) -> dict[str, float]:
        n = max(self.images_sent, 1)
        return {**asdict(self), "model_calls": self.captions_sent + self.objects_sent,
                "avg_bytes_before_scaling": round(self.bytes_before_scaling / n),
                "avg_bytes_sent": round(self.bytes_sent / n)}


@dataclass
class Request:
    key: Key
    ref: FrameRef
    ts: datetime  # event time of the trigger
    work: Callable[[bytes], Any]  # jpeg -> an event to publish, or None
    urgent: bool = False  # an enter or exit: goes before periodic work
    started: Callable[["Request"], None] | None = None  # the frame is loaded and about to go to the model
    missing: Callable[["Request"], None] | None = None  # the frame could not be loaded: nothing was sent
    tag: Any = None  # the requester's own note
    seq: int = 0


class ModelQueue:
    def __init__(self, load_keyframe: Callable[[FrameRef], bytes], min_gap_s: float = MIN_GAP_S,
                 yield_after_question_s: float = YIELD_AFTER_QUESTION_S, yield_timeout_s: float = YIELD_TIMEOUT_S) -> None:
        self.load_keyframe = load_keyframe
        self.min_gap = timedelta(seconds=min_gap_s)
        self.yield_after = timedelta(seconds=yield_after_question_s)
        self.yield_timeout = timedelta(seconds=yield_timeout_s)
        self.queued: dict[Key, Request] = {}
        self.running: Request | None = None  # the one call in flight
        self.now: datetime | None = None  # newest event time seen
        self.free_at: datetime | None = None  # end of the last call + the gap
        self.questions: dict[str, datetime] = {}  # question id -> when it was asked
        self.quiet_until: datetime | None = None  # the grace after the last answer
        self.errors = 0
        self._seq = 0

    # --- the clock and the people ---

    def tick(self, ts: datetime) -> None:
        if self.now is None or ts > self.now:
            self.now = ts

    def on_question(self, e) -> None:
        self.tick(e.ts)
        if self.yield_timeout > timedelta(0):  # 0: perception does not give way
            self.questions[e.question_id] = e.ts

    def on_answer(self, e) -> None:
        self.tick(e.ts)
        if self.questions.pop(e.question_id, None) is not None:
            until = e.ts + self.yield_after
            self.quiet_until = until if self.quiet_until is None else max(self.quiet_until, until)

    def yielding(self) -> bool:
        """A question is being answered, or was a moment ago."""
        if self.now is None:
            return False
        self.questions = {q: asked for q, asked in self.questions.items() if self.now - asked < self.yield_timeout}
        return bool(self.questions) or (self.quiet_until is not None and self.now < self.quiet_until)

    def free(self) -> bool:
        if self.running is not None or self.yielding():
            return False
        return self.free_at is None or self.now is None or self.now >= self.free_at

    # --- the queue ---

    def pending(self, key: Key) -> Request | None:
        return self.queued.get(key)

    def put(self, request: Request) -> Request | None:
        """Queue `request`, in place of an older one for the same kind and camera. Returns the one it replaced."""
        self.tick(request.ts)
        old = self.queued.get(request.key)
        if old is None:
            self._seq += 1
            request.seq = self._seq
        else:
            request.seq = old.seq  # keeps its turn: a busy camera cannot starve a quiet one
        self.queued[request.key] = request
        return old

    async def drain(self) -> list:
        """Start what the rules allow, one call at a time. Returns what the model calls produced, to publish."""
        out = []
        while self.queued and self.free():
            request = min(self.queued.values(), key=lambda r: (not r.urgent, r.seq))
            del self.queued[request.key]
            self.running = request
            try:
                jpeg = await asyncio.to_thread(self._load, request.ref)
                if jpeg is None:  # the cadence is not used up: the next frame gets its turn
                    if request.missing:
                        request.missing(request)
                    continue
                if request.started:
                    request.started(request)
                try:
                    result = await asyncio.to_thread(request.work, jpeg)
                except Exception:  # one failed call must not lose the others or stop the queue
                    self.errors += 1
                    log.exception("perception model call failed for %s", request.ref)
                    result = None
                if self.now is not None:
                    self.free_at = self.now + self.min_gap
                if result is not None:
                    out.append(result)
            finally:
                self.running = None
        return out

    def _load(self, ref: FrameRef) -> bytes | None:
        try:
            return self.load_keyframe(ref)
        except Exception as err:  # pruned, never written, or a ref from a fixture
            log.debug("no keyframe for %s: %s", ref, err)
            return None
