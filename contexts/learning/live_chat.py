"""The live chat beside the moving power graph on the Speed and energy tab: the answer's tokens stream in while the
watts move, then the joules for that answer.

    POST /api/learning/lab/live-chat      {question, target: "pair" | "box"}
    answers text/event-stream, one Server-Sent Event per line group:

    event: token   data: {"text": str}                      words as the model server streams them
    event: tool    data: {"name": str, "arguments": {...}}  a tool the model called, run on the box's memory
    event: done    data: {"answer", "target", "serving", "tokens", "seconds", "tok_s", "ttft_s", "calls",
                          "energy": {...} | null}
    event: error   data: {"error": str, "seconds": float}

The pair is the 122B brain, the head on pod C port 8000 through the hub proxy (contexts/circle/brain_pair.py's keys
and client settings), served name "teacher"; the box is llama-server on the box (runner.llm_url, 127.0.0.1:8080).
Both are asked with OpenAI chat completions, stream true, in lab_deploy's chat prompt with the memory tools offered,
not forced. A tool call in the stream runs against the box's memory with lab_deploy.run_tool (the agent's own
executor), then one more streamed turn without tools gives the answer: one round, for speed.

`tokens` is the server's completion token count when it reports usage, else the number of streamed chunks that
carried text or a tool call (vLLM and llama-server send one token per chunk). `ttft_s` is from the request to the
first streamed chunk; `tok_s` is tokens over the time between each turn's first and last chunk. `energy` is computed
the way deploy/ask does: for the pair, both pods' samplers over the request window
(BrainPair.window_energy: {joules, watts_avg, seconds, pods: {C, D}}, each pod with its idle_w beside); for the box,
the Jetson sampler (AnswerMeter.measure: {joules, idle_w, seconds, sampler, samples, above_idle_j, low_confidence})
with watts_avg added. Gross, nothing subtracted. null when the meters do not cover the window ("meters offline").

The whole answer has a TIMEOUT_S limit; any failure is an `error` event, never a broken stream.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime
from typing import Any

import httpx

from contexts.learning import lab_deploy

log = logging.getLogger("atlas.learning.live_chat")

TIMEOUT_S = 90.0
TARGETS = ("pair", "box")
BOX_SERVING = "the box, 4B"
_OPEN_RE = re.compile(r"<(think|tool_call)>")
_TAGS = ("<think>", "<tool_call>", "<|")


def sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def visible(raw: str) -> str:
    """The part of a streamed turn that is safe to show now: no thinking, no tool-call markup, no template tokens, and
    nothing after an unclosed <think> or <tool_call>, nor a trailing '<...' that could still become one."""
    text = lab_deploy.CALL_MARKUP_RE.sub("", lab_deploy.THINK_RE.sub("", raw))
    m = _OPEN_RE.search(text)
    if m:
        text = text[: m.start()]
    cut = text.rfind("<")
    if cut >= 0 and ">" not in text[cut:] and any(tag.startswith(text[cut:]) or text[cut:].startswith("<|") for tag in _TAGS):
        text = text[:cut]
    text = lab_deploy.SPECIAL_RE.sub("", text.split(lab_deploy.IM_END)[0])
    return text.lstrip()


async def _chunks(response: httpx.Response) -> AsyncIterator[dict[str, Any]]:
    """The JSON chunks of an OpenAI streaming response (data: lines), up to [DONE]."""
    async for line in response.aiter_lines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            return
        try:
            chunk = json.loads(payload)
        except ValueError:
            continue
        if isinstance(chunk, dict):
            yield chunk


class Turn:
    """One streamed model turn, folded: its raw text, the tool calls assembled from deltas, the token count."""

    def __init__(self) -> None:
        self.raw = ""
        self.shown = ""
        self.calls: dict[int, dict[str, Any]] = {}
        self.chunks = 0
        self.usage: int | None = None
        self.first: float | None = None
        self.last: float | None = None

    def message(self) -> dict[str, Any]:
        tool_calls = [{"id": c.get("id") or f"call-{i}", "type": "function",
                       "function": {"name": c.get("name") or "", "arguments": c.get("arguments") or ""}}
                      for i, c in sorted(self.calls.items())]
        return {"content": self.raw, "tool_calls": tool_calls}

    @property
    def tokens(self) -> int:
        return self.usage if self.usage is not None else self.chunks


async def stream_turn(open_stream: Callable[[dict[str, Any]], Any], body: dict[str, Any],
                      emit: Callable[[str, dict[str, Any]], Any], clock: Callable[[], float]) -> Turn:
    """Streams one turn, emitting each newly visible piece of text as a `token` event as it arrives."""
    turn = Turn()
    async with open_stream(body) as response:
        if response.status_code != 200:
            raise RuntimeError(f"the model server answered HTTP {response.status_code}")
        async for chunk in _chunks(response):
            usage = chunk.get("usage")
            if isinstance(usage, dict) and isinstance(usage.get("completion_tokens"), int):
                turn.usage = usage["completion_tokens"]
            choices = chunk.get("choices") or []
            if not choices or not isinstance(choices[0], dict):
                continue
            delta = choices[0].get("delta") or {}
            text = delta.get("content") or ""
            calls = delta.get("tool_calls") or []
            if not text and not calls:
                continue
            now = clock()
            turn.first = now if turn.first is None else turn.first
            turn.last = now
            turn.chunks += 1
            for call in calls:
                slot = turn.calls.setdefault(int(call.get("index", 0) or 0), {})
                if call.get("id"):
                    slot["id"] = call["id"]
                fn = call.get("function") or {}
                if fn.get("name"):
                    slot["name"] = fn["name"]
                if fn.get("arguments"):
                    slot["arguments"] = (slot.get("arguments") or "") + str(fn["arguments"])
            if text:
                turn.raw += str(text)
                now_shown = visible(turn.raw)
                if len(now_shown) > len(turn.shown) and now_shown.startswith(turn.shown):
                    await emit("token", {"text": now_shown[len(turn.shown):]})
                    turn.shown = now_shown
    return turn


def _pair_opener(pair: Any) -> Callable[[dict[str, Any]], Any]:
    """Streams from the head through the hub proxy, with the pair's keys and client settings (brain_pair.chat)."""
    from contexts.circle.brain_pair import MODEL_PORT

    head = pair.head
    if head.base is None:
        raise RuntimeError(getattr(pair, "reason", None) or "this box has no keys for the head pod")
    client = httpx.AsyncClient(transport=getattr(pair, "transport", None), timeout=TIMEOUT_S, follow_redirects=False,
                               headers={"User-Agent": getattr(pair, "user_agent", "atlas")})

    def open_stream(body: dict[str, Any]):
        return client.stream("POST", f"{head.base}/{MODEL_PORT}/v1/chat/completions",
                             json={**body, "model": lab_deploy.BRAIN_MODEL,
                                   "chat_template_kwargs": {"enable_thinking": False}},
                             headers={"Authorization": f"token {head.token}"})

    open_stream.client = client  # type: ignore[attr-defined]
    return open_stream


def _box_opener(llm_url: str, transport: httpx.AsyncBaseTransport | None) -> Callable[[dict[str, Any]], Any]:
    client = httpx.AsyncClient(base_url=llm_url.rstrip("/"), transport=transport, timeout=TIMEOUT_S)

    def open_stream(body: dict[str, Any]):
        return client.stream("POST", "/v1/chat/completions", json=body)

    open_stream.client = client  # type: ignore[attr-defined]
    return open_stream


async def run_chat(question: str, target: str, *, emit: Callable[[str, dict[str, Any]], Any], qa: Any,
                   pair: Any | None = None, llm_url: str = lab_deploy.DEFAULT_LLM_URL,
                   transport: httpx.AsyncBaseTransport | None = None, meter: Any | None = None,
                   clock: Callable[[], float] = time.time) -> dict[str, Any]:
    """Asks the target with streaming, emits tokens and tool calls, and returns the `done` event's body."""
    if target == "pair":
        if pair is None:
            raise RuntimeError("this box has no keys for the pods (see contexts/circle/README.md)")
        opener = _pair_opener(pair)
    else:
        opener = _box_opener(llm_url, transport)
    began = clock()
    started = datetime.now(UTC)
    base = {"temperature": 0.2, "max_tokens": lab_deploy.ANSWER_TOKENS, "stream": True,
            "stream_options": {"include_usage": True}}
    messages = [{"role": "system", "content": lab_deploy.chat_system(datetime.now().astimezone())},
                {"role": "user", "content": question}]
    turns: list[Turn] = []
    calls: list[dict[str, Any]] = []
    try:
        first = await stream_turn(opener, {**base, "messages": messages, "tools": lab_deploy._memory_tools(),
                                           "tool_choice": "auto"}, emit, clock)
        turns.append(first)
        found = lab_deploy._calls_of(first.message())
        if found:  # one round: run the calls on the box's memory, then stream the answer without tools
            messages.append({"role": "assistant", "content": first.shown,
                             "tool_calls": [{"id": c["id"], "type": "function", "function": {
                                 "name": c["name"], "arguments": json.dumps(c["arguments"])}} for c in found]})
            for c in found:
                calls.append({"name": c["name"], "arguments": c["arguments"]})
                await emit("tool", {"name": c["name"], "arguments": c["arguments"]})
                messages.append({"role": "tool", "tool_call_id": c["id"],
                                 "content": await lab_deploy.run_tool(qa, c["name"], c["arguments"])})
            turns.append(await stream_turn(opener, {**base, "messages": messages}, emit, clock))
    finally:
        await opener.client.aclose()  # type: ignore[attr-defined]
    ended_at = clock()
    ended = datetime.now(UTC)
    seconds = max(ended_at - began, 1e-9)
    tokens = sum(t.tokens for t in turns)
    gen = sum((t.last - t.first) for t in turns if t.first is not None and t.last is not None)
    firsts = [t.first for t in turns if t.first is not None]
    answer = " ".join(" ".join(t.shown for t in turns).split())
    energy = None
    try:
        if target == "pair":
            energy = await pair.window_energy(began, ended_at)
        elif meter is not None:
            got = await meter.measure(started, ended)
            if got and isinstance(got.get("joules"), int | float) and got.get("seconds"):
                energy = {**got, "watts_avg": round(got["joules"] / max(float(got["seconds"]), 1e-9), 1), "pods": None}
            else:
                energy = got
    except Exception:
        log.exception("measuring the live chat answer failed")
        energy = None
    return {"answer": answer or "(no answer)", "target": target,
            "serving": lab_deploy.BRAIN_SERVING if target == "pair" else BOX_SERVING,
            "tokens": tokens, "seconds": round(seconds, 2),
            "tok_s": round(tokens / gen, 1) if gen > 0 and tokens > 1 else None,
            "ttft_s": round(firsts[0] - began, 2) if firsts else None, "calls": calls, "energy": energy}


async def event_stream(question: str, target: str, *, timeout_s: float = TIMEOUT_S, **kwargs: Any
                       ) -> AsyncIterator[str]:
    """The SSE body: the chat runs as a task feeding a queue, so the timeout covers the whole answer."""
    queue: asyncio.Queue[tuple[str, dict[str, Any]] | None] = asyncio.Queue()
    began = time.monotonic()

    async def emit(event: str, data: dict[str, Any]) -> None:
        await queue.put((event, data))

    async def work() -> None:
        try:
            done = await run_chat(question, target, emit=emit, **kwargs)
            await queue.put(("done", done))
        except Exception as exc:  # said in the stream, never raised into it
            text = str(exc)[:200] or type(exc).__name__
            if isinstance(exc, httpx.HTTPError):
                text = f"the model server did not answer ({type(exc).__name__})"
            await queue.put(("error", {"error": text, "seconds": round(time.monotonic() - began, 1)}))
        finally:
            await queue.put(None)

    task = asyncio.create_task(work())
    try:
        while True:
            left = timeout_s - (time.monotonic() - began)
            try:
                item = await asyncio.wait_for(queue.get(), max(left, 0.01))
            except (asyncio.TimeoutError, TimeoutError):
                yield sse("error", {"error": f"no answer in {timeout_s:.0f} s", "seconds": round(time.monotonic() - began, 1)})
                return
            if item is None:
                return
            yield sse(*item)
    finally:
        if not task.done():
            task.cancel()
