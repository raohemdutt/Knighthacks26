"""The device GPU scheduler (R4): who gets the Atlas model next.

llama-server on the Jetson has a fixed number of slots (`-np`, 1 today), and perception, Q&A, summaries,
the day narrator and SPOT all share them. This module decides the order, in-process, for every caller
that goes through one runtime instance:

- **Priority** (Design Doc, Context: Inference Runtime): detection > alerts > chat > captions >
  summaries > backfill. A waiting request of a higher rank always goes before a lower one; within a rank,
  first come first served (the old two-level gate let whichever thread woke first win).
- **VLM rate cap**: background image requests (captions, object lists, open-vocabulary detection at
  caption rank or below) are limited to `vlm_max_per_min` starts per rolling minute. A capped request
  waits *without* holding a slot, so text work behind it (summaries) still runs. Chat is never capped.
- **Batching**: with `slots` > 1 (llama-server `-np` > 1) several requests run at once and llama.cpp
  decodes them in one batch. Background work may hold at most `slots - 1` of them, so a question never
  waits for a full house of captions. Identical requests that arrive while one is in flight are
  coalesced: they share the one answer instead of running twice (works at `-np 1` too).

Detection (TensorRT) and alerts (pure rules) do not call llama-server today; their ranks exist so that
anything on those paths that ever does is served first.
"""

import contextlib
import itertools
import threading
import time
from collections import deque
from collections.abc import Callable, Hashable
from typing import Any

RANK: dict[str, int] = {"detection": 0, "alert": 1, "chat": 2, "caption": 3, "summary": 4, "backfill": 5}
BACKGROUND = RANK["caption"]  # this rank and below: captions, summaries, backfill


def rank_of(priority: str) -> int:
    try:
        return RANK[priority]
    except KeyError:
        raise ValueError(f"unknown priority {priority!r}; one of {list(RANK)}") from None


class GpuScheduler:
    def __init__(
        self,
        slots: int = 1,
        vlm_max_per_min: int | None = None,
        coalesce: bool = True,
        clock: Callable[[], float] = time.monotonic,
        window_s: float = 60.0,
    ) -> None:
        if slots < 1:
            raise ValueError("slots must be at least 1")
        self.slots = slots
        self.vlm_max = vlm_max_per_min
        self.window_s = window_s
        self.coalesce_enabled = coalesce
        self.clock = clock
        self._cv = threading.Condition()
        self._seq = itertools.count()
        self._waiting: dict[int, tuple[int, bool]] = {}  # ticket -> (rank, rate-capped)
        self._busy = 0
        self._busy_background = 0
        self._vlm_starts: deque[float] = deque()
        self._inflight: dict[Hashable, _Shared] = {}
        self.counts = {"served": 0, "coalesced": 0, "capped_waits": 0}

    @classmethod
    def from_config(cls, cfg: dict | None) -> "GpuScheduler":
        """models.yaml `scheduler:` block; missing keys keep the defaults."""
        cfg = cfg or {}
        cap = cfg.get("vlm_max_per_min")
        return cls(slots=int(cfg.get("slots", 1)), vlm_max_per_min=int(cap) if cap else None,
                   coalesce=bool(cfg.get("coalesce", True)))

    # -- slots -------------------------------------------------------------------------------------

    @contextlib.contextmanager
    def slot(self, priority: str = "caption", vlm: bool = False):
        """Hold one llama-server slot for the body. `vlm`: the request carries an image."""
        rank = rank_of(priority)
        background = rank >= BACKGROUND
        capped = vlm and background and self.vlm_max is not None
        with self._cv:
            ticket = next(self._seq)
            self._waiting[ticket] = (rank, capped)
            noted_cap = False
            try:
                while True:
                    now = self.clock()
                    if self._winner(now) == ticket:
                        break
                    if capped and not noted_cap and not self._cap_free(now):
                        self.counts["capped_waits"] += 1
                        noted_cap = True
                    self._cv.wait(self._wake_in(now))
            except BaseException:
                del self._waiting[ticket]
                self._cv.notify_all()
                raise
            del self._waiting[ticket]
            self._busy += 1
            self._busy_background += background
            if capped:
                self._vlm_starts.append(self.clock())
            self.counts["served"] += 1
            self._cv.notify_all()  # with spare slots the next winner can start too
        try:
            yield
        finally:
            with self._cv:
                self._busy -= 1
                self._busy_background -= background
                self._cv.notify_all()

    def _winner(self, now: float) -> int | None:
        """The ticket that may start now, or None (all slots busy, or nothing eligible)."""
        if self._busy >= self.slots:
            return None
        background_room = self._busy_background < max(1, self.slots - 1)
        cap_free = self._cap_free(now)
        best = None
        for ticket, (rank, capped) in self._waiting.items():
            if rank >= BACKGROUND and not background_room:
                continue
            if capped and not cap_free:
                continue
            if best is None or (rank, ticket) < best:
                best = (rank, ticket)
        return best[1] if best else None

    def _cap_free(self, now: float) -> bool:
        if self.vlm_max is None:
            return True
        while self._vlm_starts and now - self._vlm_starts[0] >= self.window_s:
            self._vlm_starts.popleft()
        return len(self._vlm_starts) < self.vlm_max

    def _wake_in(self, now: float) -> float | None:
        """When a capped request is waiting on the cap, wake when the oldest start leaves the window."""
        if self._vlm_starts and any(c for _, c in self._waiting.values()) and not self._cap_free(now):
            return min(max(0.0, self._vlm_starts[0] + self.window_s - now) + 1e-3, 0.5)  # re-check: clocks can jump
        return None

    # -- coalescing --------------------------------------------------------------------------------

    def coalesce(self, key: Hashable, run: Callable[[], Any]) -> Any:
        """Run `run()` once per key at a time: callers with the same key while it runs get its result."""
        if not self.coalesce_enabled:
            return run()
        with self._cv:
            shared = self._inflight.get(key)
            leader = shared is None
            if leader:
                shared = self._inflight[key] = _Shared()
            else:
                self.counts["coalesced"] += 1
        if not leader:
            return shared.wait()
        try:
            shared.result = run()
        except BaseException as e:
            shared.error = e
            raise
        finally:
            with self._cv:
                del self._inflight[key]
            shared.done.set()
        return shared.result

    def stats(self) -> dict:
        with self._cv:
            waiting = {p: 0 for p in RANK}
            names = {r: p for p, r in RANK.items()}
            for rank, _ in self._waiting.values():
                waiting[names[rank]] += 1
            return {"slots": self.slots, "busy": self._busy, "waiting": waiting,
                    "vlm_max_per_min": self.vlm_max, "vlm_starts_in_window": len(self._vlm_starts), **self.counts}


class _Shared:
    def __init__(self) -> None:
        self.done = threading.Event()
        self.result: Any = None
        self.error: BaseException | None = None

    def wait(self) -> Any:
        self.done.wait()
        if self.error is not None:
            raise self.error
        return self.result
