"""The "16 people asking" button on the live power graph: `streams` people asking the 122B pair at once for about
`seconds`, so the graph shows the pods' watts while the pair serves them all together.

    POST /api/learning/lab/load   {streams: 16, seconds: 30}                                   (login, like deploy)
                               or {ramp: [1, 4, 16, 32], seconds_per_stage: 20}             (the "Ramp 1 to 32" button)
        202 the view below, state "running"
        400 a bad body; 409 a load run already runs; 503 the pair client is unavailable here (the reason)
    GET  /api/learning/lab/load
        {"available": bool, "reason": str | null,
         "state": "idle" | "running" | "done" | "failed", "streams": int | null, "seconds": int | null,
         "started": iso | null, "ended": iso | null, "started_ms": int | null, "ended_ms": int | null,
         "requests": int, "errors": int, "tokens": int, "tok_s_total": float | null, "tok_s_per_stream": float | null,
         "ttft_s": float | null,
         "energy": {"joules", "watts_avg", "seconds", "pods": {"C": {...}, "D": {...}}} | null,
         "j_per_token": float | null, "message": str | null,
         "ramp": [int] | null, "seconds_per_stage": int | null, "stage": int | null (1-based, while running),
         "stages": [{"streams", "requests", "tokens", "tok_s_total", "watts_avg", "j_per_token", "started_ms",
                     "ended_ms"}]}

A ramp never stops a stream: at each stage's start it ADDS streams up to that stage's count (1, then 3 more, then 12
more, then 16 more), so the load never drops between stages. Stage boundaries are timestamps only: a stage's row
counts the answers that finished inside it, and is filled the moment the stage ends (its watts and J per token once
the run is over, from the samplers over that stage's window). A plain {streams, seconds} run is a one-stage ramp.

Only the pair is asked (the head pod's port 8000 through the hub proxy, contexts/circle/brain_pair.py's keys), never
the box's 4B. Each stream loops: a short plain question about a desk from QUESTIONS, no tools, max_tokens MAX_TOKENS,
thinking off, streamed so its time to first token is measured, then the next question, until `seconds` have passed (a
request already sent finishes). All streams run at once on one event loop in a background thread.

`requests` counts completed answers, `errors` failed ones. `tokens` is the server's completion token count when it
reports usage, else the streamed chunks. `tok_s_total` is tokens over the run's wall time (first request sent to last
answer done), `tok_s_per_stream` that over `streams`, `ttft_s` the median time to first token over the answers.
`energy` is both pods' samplers over the run window (BrainPair.window_energy: gross joules, nothing subtracted), null
when the meters do not cover it; `j_per_token` is its joules over `tokens`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import statistics
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import httpx

log = logging.getLogger("atlas.learning.load_run")

DEFAULT_STREAMS, MAX_STREAMS = 16, 32  # the pair serves at most 32 sequences at once
DEFAULT_SECONDS = 30
MIN_SECONDS, MAX_SECONDS = 10, 120
MAX_TOKENS = 96
MIN_STAGE_S, MAX_STAGE_S, DEFAULT_STAGE_S = 10, 60, 20
MAX_STAGES = 6
REQUEST_TIMEOUT_S = 90.0
AVAILABLE_CACHE_S = 30.0
QUESTIONS = (
    "What is a desk usually made of?",
    "How tall is a typical desk?",
    "Why do desks have drawers?",
    "What is a standing desk?",
    "How do I keep my desk tidy?",
    "What should sit on a desk for good posture?",
    "Where is the best place in a room for a desk?",
    "What is the difference between a desk and a table?",
    "How do I clean a wooden desk?",
    "What lamp works well on a desk?",
    "How deep should a desk be for two monitors?",
    "Why do some desks have a slanted top?",
)


class Refused(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


def _iso(t: float | None) -> str | None:
    return datetime.fromtimestamp(t, UTC).isoformat().replace("+00:00", "Z") if t is not None else None


def default_pair() -> Any:
    """A pair built from the box's .env for this one run (its async clients live and die on the run's own loop)."""
    from contexts.circle.brain_pair import build_brain_pair

    return build_brain_pair()


def pair_available(build: Callable[[], Any] = default_pair) -> str | None:
    """None when the pair client can be built with the head's keys, else why not (key names only)."""
    try:
        pair = build()
    except Exception as exc:
        return f"the pair client could not be built here ({type(exc).__name__}: {str(exc)[:120]})"
    if pair is None or not getattr(pair, "configured", False):
        return getattr(pair, "reason", None) or "this box has no keys for the head pod"
    return None


async def _window_energy(pair: Any, start: float, end: float) -> dict[str, Any] | None:
    return await pair.window_energy(start, end)


class LoadRunner:
    """At most one load run at a time. build_pair, energy, available, clock and monotonic replace the pods, the
    samplers, the key check and the time in tests."""

    def __init__(self, *, build_pair: Callable[[], Any] = default_pair,
                 energy: Callable[[Any, float, float], Any] = _window_energy,
                 available: Callable[[], str | None] | None = None,
                 clock: Callable[[], float] = time.time, monotonic: Callable[[], float] = time.monotonic) -> None:
        self.build_pair = build_pair
        self.energy = energy
        self.available = available if available is not None else (lambda: pair_available(build_pair))
        self.clock = clock
        self.monotonic = monotonic
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._state: dict[str, Any] = self._idle()
        self._reason: tuple[float, str | None] | None = None

    def _available(self, fresh: bool = False) -> str | None:
        now = time.monotonic()
        if fresh or self._reason is None or now - self._reason[0] > AVAILABLE_CACHE_S:
            self._reason = (now, self.available())
        return self._reason[1]

    @staticmethod
    def _idle() -> dict[str, Any]:
        return {"state": "idle", "streams": None, "seconds": None, "started": None, "ended": None, "requests": 0,
                "errors": 0, "tokens": 0, "tok_s_total": None, "tok_s_per_stream": None, "ttft_s": None,
                "energy": None, "j_per_token": None, "message": None, "ramp": None, "seconds_per_stage": None,
                "stage": None, "stages": []}

    def view(self) -> dict[str, Any]:
        with self._lock:
            s = json.loads(json.dumps(self._state))
        reason = None if s["state"] == "running" else self._available()
        started, ended = s.pop("_t0", None), s.pop("_t1", None)
        return {"available": reason is None, "reason": reason, **s,
                "started_ms": round(started * 1000) if started is not None else None,
                "ended_ms": round(ended * 1000) if ended is not None else None}

    def start(self, streams: Any, seconds: Any, ramp: Any = None, seconds_per_stage: Any = None) -> dict[str, Any]:
        if ramp is not None:
            ok = (isinstance(ramp, list) and 1 <= len(ramp) <= MAX_STAGES
                  and all(isinstance(n, int) and not isinstance(n, bool) and 1 <= n <= MAX_STREAMS for n in ramp)
                  and all(a < b for a, b in zip(ramp, ramp[1:], strict=False)))
            if not ok:
                raise Refused(400, f"ramp is a rising list of stream counts from 1 to {MAX_STREAMS}, e.g. [1, 4, 16, 32]")
            if seconds_per_stage is None:
                seconds_per_stage = DEFAULT_STAGE_S
            if (isinstance(seconds_per_stage, bool) or not isinstance(seconds_per_stage, int | float)
                    or not MIN_STAGE_S <= seconds_per_stage <= MAX_STAGE_S):
                raise Refused(400, f"seconds_per_stage is a number from {MIN_STAGE_S} to {MAX_STAGE_S}")
            return self._begin(list(ramp), int(seconds_per_stage), is_ramp=True)
        if streams is None:
            streams = DEFAULT_STREAMS
        if isinstance(streams, bool) or not isinstance(streams, int) or not 1 <= streams <= MAX_STREAMS:
            raise Refused(400, f"streams is a whole number from 1 to {MAX_STREAMS}")
        if seconds is None:
            seconds = DEFAULT_SECONDS
        if isinstance(seconds, bool) or not isinstance(seconds, int | float) or not MIN_SECONDS <= seconds <= MAX_SECONDS:
            raise Refused(400, f"seconds is a number from {MIN_SECONDS} to {MAX_SECONDS}")
        return self._begin([streams], int(seconds), is_ramp=False)

    def _begin(self, plan: list[int], stage_s: int, *, is_ramp: bool) -> dict[str, Any]:
        streams, seconds = plan[-1], stage_s * len(plan)
        with self._lock:
            if self._state["state"] == "running":
                raise Refused(409, f"a load run of {self._state['streams']} streams is already running")
        reason = self._available(fresh=True)
        if reason:
            raise Refused(503, reason)
        with self._lock:
            if self._state["state"] == "running":
                raise Refused(409, "a load run is already running")
            t0 = self.clock()
            self._state = {**self._idle(), "state": "running", "streams": streams, "seconds": seconds,
                           "started": _iso(t0), "_t0": t0, "_t1": None, "ramp": plan if is_ramp else None,
                           "seconds_per_stage": stage_s if is_ramp else None, "stage": 1}
            self._thread = threading.Thread(target=self._run, args=(plan, stage_s, t0), daemon=True,
                                            name="load-run")
            self._thread.start()
        return self.view()

    def _run(self, plan: list[int], stage_s: int, t0: float) -> None:
        try:
            asyncio.run(self._drive(plan, stage_s, t0))
        except Exception as exc:
            log.exception("the load run failed")
            with self._lock:
                t1 = self.clock()
                self._state.update({"state": "failed", "ended": _iso(t1), "_t1": t1,
                                    "stage": None, "message": f"{type(exc).__name__}: {str(exc)[:160]}"})

    async def _ask(self, client: httpx.AsyncClient, url: str, token: str | None, question: str,
                   tally: dict[str, Any]) -> None:
        body = {"model": _brain_model(), "messages": [{"role": "user", "content": question}],
                "max_tokens": MAX_TOKENS, "temperature": 0.7, "stream": True,
                "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False}}
        sent = self.monotonic()
        first: float | None = None
        chunks = 0
        usage: int | None = None
        async with client.stream("POST", url, json=body, headers={"Authorization": f"token {token}"}) as reply:
            if reply.status_code != 200:
                raise RuntimeError(f"the head pod answered HTTP {reply.status_code}")
            async for line in reply.aiter_lines():
                line = line.strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except ValueError:
                    continue
                if not isinstance(chunk, dict):
                    continue
                got = chunk.get("usage")
                if isinstance(got, dict) and isinstance(got.get("completion_tokens"), int):
                    usage = got["completion_tokens"]
                choices = chunk.get("choices") or []
                delta = (choices[0].get("delta") or {}) if choices and isinstance(choices[0], dict) else {}
                if delta.get("content"):
                    chunks += 1
                    if first is None:
                        first = self.monotonic()
        got_tokens = usage if usage is not None else chunks
        tally["requests"] += 1
        tally["tokens"] += got_tokens
        stage = tally["per_stage"][tally["stage"]]  # the stage the answer finished in
        stage["requests"] += 1
        stage["tokens"] += got_tokens
        if first is not None:
            tally["ttft"].append(first - sent)

    async def _stream(self, n: int, client: httpx.AsyncClient, url: str, token: str | None, deadline: float,
                      tally: dict[str, Any]) -> None:
        i = n
        while self.monotonic() < deadline:
            question = QUESTIONS[i % len(QUESTIONS)]
            i += 1
            try:
                await self._ask(client, url, token, question, tally)
            except Exception as exc:
                tally["errors"] += 1
                tally["last_error"] = f"{type(exc).__name__}: {str(exc)[:160]}"
                await asyncio.sleep(0.5)  # a refusing server is not hammered in a tight loop
            with self._lock:
                self._state.update({"requests": tally["requests"], "errors": tally["errors"],
                                    "tokens": tally["tokens"]})

    async def _drive(self, plan: list[int], stage_s: int, t0: float) -> None:
        from contexts.circle.brain_pair import MODEL_PORT

        pair = self.build_pair()
        tally: dict[str, Any] = {"requests": 0, "errors": 0, "tokens": 0, "ttft": [], "last_error": None,
                                 "stage": 0, "per_stage": [{"requests": 0, "tokens": 0} for _ in plan]}
        rows: list[dict[str, Any]] = []
        bounds: list[float] = []
        streams = plan[-1]
        try:
            head = pair.head
            url = f"{head.base}/{MODEL_PORT}/v1/chat/completions"
            limits = httpx.Limits(max_connections=streams + 4, max_keepalive_connections=streams + 4)
            async with httpx.AsyncClient(transport=getattr(pair, "transport", None), timeout=REQUEST_TIMEOUT_S,
                                         follow_redirects=False, limits=limits,
                                         headers={"User-Agent": getattr(pair, "user_agent", "atlas")}) as client:
                base = self.monotonic()
                deadline = base + stage_s * len(plan)
                tasks: list[asyncio.Task] = []
                for i, target in enumerate(plan):
                    tally["stage"] = i
                    bounds.append(t0 if i == 0 else self.clock())
                    with self._lock:
                        self._state["stage"] = i + 1
                    for n in range(len(tasks), target):  # add streams; the running ones keep looping
                        tasks.append(asyncio.create_task(self._stream(n, client, url, head.token, deadline, tally)))
                    if i < len(plan) - 1:
                        stage_end = base + stage_s * (i + 1)
                        while self.monotonic() < stage_end:
                            await asyncio.sleep(0.02)
                        rows.append(self._row(plan[i], tally["per_stage"][i], bounds[i], self.clock()))
                        with self._lock:
                            self._state["stages"] = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
                await asyncio.gather(*tasks)
            t1 = self.clock()
            rows.append(self._row(plan[-1], tally["per_stage"][-1], bounds[-1], t1))
            energy = None
            try:
                energy = await self.energy(pair, t0, t1)
            except Exception:
                log.exception("measuring the load run failed")
            if len(plan) > 1 and energy is not None:
                for row in rows:
                    try:
                        got = await self.energy(pair, row["_s"], row["_e"])
                    except Exception:
                        got = None
                    self._fill(row, got)
            elif energy is not None:
                self._fill(rows[0], energy)
        finally:
            closer = getattr(pair, "aclose", None)
            if closer is not None:
                await closer()
        wall = max(t1 - t0, 1e-9)
        tokens = tally["tokens"]
        tok_s = round(tokens / wall, 1) if tokens else None
        joules = energy.get("joules") if isinstance(energy, dict) else None
        message = None
        if tally["errors"]:
            message = f"{tally['errors']} request(s) failed; the last: {tally['last_error']}"
        for row in rows:
            row.pop("_s", None)
            row.pop("_e", None)
        with self._lock:
            self._state.update({
                "state": "done" if tally["requests"] else "failed", "ended": _iso(t1), "_t1": t1, "stage": None,
                "requests": tally["requests"], "errors": tally["errors"], "tokens": tokens,
                "tok_s_total": tok_s, "tok_s_per_stream": round(tok_s / streams, 2) if tok_s is not None else None,
                "ttft_s": round(statistics.median(tally["ttft"]), 3) if tally["ttft"] else None,
                "energy": energy,
                "j_per_token": round(joules / tokens, 3) if isinstance(joules, int | float) and tokens else None,
                "message": message, "stages": rows})

    @staticmethod
    def _row(streams: int, got: dict[str, int], start: float, end: float) -> dict[str, Any]:
        wall = max(end - start, 1e-9)
        return {"streams": streams, "requests": got["requests"], "tokens": got["tokens"],
                "tok_s_total": round(got["tokens"] / wall, 1) if got["tokens"] else None,
                "watts_avg": None, "j_per_token": None, "started_ms": round(start * 1000),
                "ended_ms": round(end * 1000), "_s": start, "_e": end}

    @staticmethod
    def _fill(row: dict[str, Any], energy: dict[str, Any] | None) -> None:
        if not isinstance(energy, dict):
            return
        row["watts_avg"] = energy.get("watts_avg")
        joules = energy.get("joules")
        row["j_per_token"] = round(joules / row["tokens"], 3) if isinstance(joules, int | float) and row["tokens"] else None

    def wait(self, timeout: float | None = None) -> None:
        """For tests: blocks until the running load run has ended."""
        thread = self._thread
        if thread is not None:
            thread.join(timeout)


def _brain_model() -> str:
    from contexts.learning.lab_deploy import BRAIN_MODEL

    return BRAIN_MODEL
