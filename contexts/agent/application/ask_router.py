"""Ask Atlas's router: the model decides which tool answers, and the network brain answers when the box's cannot.

Every question to POST /api/ask (the Ask Atlas tab) and /api/ask/stream (the box's voice) comes here first:

1. The box's model (the 4B on llama-server) gets the question and the product tools (product_tools.TOOLS), tool
   choice left to it. It picks a tool, or says NONE.
2. It picked one: the tool runs, and the answer is the tool's own words. A tool with an outside effect returns an
   approval card (agent.action.proposed), published here; nothing is written until the phone approves (W2).
3. It picked none, but the question names something only a tool can do (the calendar, a reminder, the cameras, the
   room map, the box's model ...): the same question goes to the network brain (the teacher on the lab's pod, a large
   model that handles tools well), when this box is configured to reach it. If that is off or does not answer, the
   box's own rules pick the tool from the words (`rules`), and the answer says so.
4. Anything else (small talk, "where are my keys?", a recap, the web) returns None, and QA answers exactly as before.

Which model answered is logged and carried on the answer (`answered_by`): "the box", "the network brain", or
"the box's rules". Model calls are live only; nothing here runs on replay.
"""

import asyncio
import json
import logging
import os
import re
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import httpx

from contexts.agent.application.product_tools import Ports, ToolRegistry, ToolResult
from contracts.types import ChatMessage, ToolCall
from runtime.interface import Runtime

log = logging.getLogger(__name__)

BOX = "the box"
NETWORK = "the network brain"
RULES = "the box's rules"

BOX_CHOICE_TIMEOUT_S = 25.0  # the 4B on the Jetson: prompt plus a short tool call
NETWORK_TIMEOUT_S = 25.0
TEACHER_PORT = 8000
TEACHER_MODEL = "teacher"
HUB_USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/130.0 Safari/537.36")  # the hub's proxy turns away a bare client (lab/hub.py USER_AGENT)

SYSTEM = (
    "You are Atlas, a home assistant box with cameras. Call the one tool that does what the person asks: "
    "the calendar (list, add, find), reminders, what the cameras saw, the room map, who is here, the cameras, "
    "the box's model, energy, the learning lab, friends, the plant floor inspection (SPOT), or opening a screen. "
    "If no tool fits (small talk, a general or web question), reply with only the word NONE."
)

# The question names something only a tool can do: when the box's model chose no tool, it is not left to small talk.
NAMES_A_TOOL = re.compile(
    r"\b(?:calendar|schedule|meeting|appointment|event|remind(?:er|ers)?|camera|cameras|room map|map of the room|"
    r"the map|room plan|model|energy|joules?|watts?|learning lab|lab run|training run|shared brain|who(?:'s| is) here|"
    r"who(?:'s| is) in the room|open the|go to the|show me the|approvals?|plant floor|inspection|how is spot|"
    r"spot (?:doing|robot))\b", re.I)

# Tool calls written into the text instead of the tool_calls field (a vLLM server without a tool parser, or a small
# model): <tool_call>{"name": ..., "arguments": {...}}</tool_call>, or a bare {"name": ..., "arguments": ...}.
MARKUP = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
BARE = re.compile(r"^\s*(\{\s*\"name\"\s*:.*\})\s*$", re.S)


def calls_in_text(text: str) -> list[ToolCall]:
    out = []
    found = MARKUP.findall(text or "") or BARE.findall(text or "")
    for i, raw in enumerate(found):
        try:
            d = json.loads(raw)
            args = d.get("arguments") or d.get("parameters") or {}
            if isinstance(args, str):
                args = json.loads(args or "{}")
            out.append(ToolCall(id=f"text-{i}", name=str(d["name"]), arguments=args if isinstance(args, dict) else {}))
        except (ValueError, KeyError, TypeError):
            continue
    return out


# --- the box's rules: the last resort, only after a model chose nothing -------------------------------------------

TITLE_CALLED = re.compile(r"\b(?:called|titled|named)\s+[\"']?(?P<t>[^\"'?.!]+)", re.I)
WITH_WHOM = re.compile(r"\bwith\s+(?P<who>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)")
ADD = re.compile(r"\b(?:add|schedule|put|book|create|set up|make)\b.*\b(?:meeting|event|appointment|call|calendar|"
                 r"lunch|dinner|session|class)\b", re.I)
REMIND = re.compile(r"\bremind(?:er)?\b", re.I)
REMIND_WHAT = re.compile(r"\b(?:remind me|reminder)\s+(?:to\s+|about\s+|that\s+)?(?P<t>.+?)(?=\s+(?:at|on|in|by|tomorrow|today|tonight|this)\b|[?.!]*$)", re.I)
CAMERA_OFF = re.compile(r"\b(?:turn off|switch off|pause|stop|disable|shut off)\b.*\bcameras?\b|\bcameras?\b.*\b(?:off|pause)\b", re.I)
CAMERA_ON = re.compile(r"\b(?:turn on|switch on|resume|start|enable|unpause)\b.*\bcameras?\b|\bcameras?\b.*\bback on\b", re.I)
CAMERA_Q = re.compile(r"\bcameras?\b", re.I)
ROOM = re.compile(r"\b(?:room map|map of the room|room plan|show me the (?:room|map))\b", re.I)
WHO = re.compile(r"\bwho(?:'s| is)\s+(?:here|home|in the room|around)\b", re.I)
MODEL = re.compile(r"\b(?:which|what)\b.*\bmodel\b|\bmodel\b.*\b(?:running|serving|now|deployed)\b", re.I)
ENERGY = re.compile(r"\b(?:energy|joules?|watts?|power)\b", re.I)
LAB = re.compile(r"\b(?:learning lab|lab run|training run|the lab|learning status|fine-?tun)", re.I)
BRAIN = re.compile(r"\bshared brain\b", re.I)
CALENDAR = re.compile(r"\b(?:calendar|schedule|agenda)\b", re.I)
DAY_WORD = re.compile(r"\b(?:today|tomorrow|yesterday|monday|tuesday|wednesday|thursday|friday|saturday|sunday|\d{4}-\d{2}-\d{2})\b", re.I)
RECAP = re.compile(r"\bwhat (?:did i do|happened|did you see)\b.*\b(?P<part>this morning|this afternoon|this evening|today|last hour|yesterday)\b", re.I)
# Inspection (contexts/spot): asked after every older rule, so no question an older rule answered changes its tool
INSPECTION = re.compile(r"\b(?:plant floor|inspection|how is spot|spot (?:doing|robot))\b", re.I)
OPEN = re.compile(r"\b(?:open|go to|show me|take me to)\s+(?:the\s+|my\s+)?(?P<s>[a-z ]+?)(?:\s+(?:screen|page|tab))?[?.!]*$", re.I)


def rules(question: str) -> ToolCall | None:
    """The tool the words name, with its arguments, or None. No model: the last resort."""
    q = question.strip()

    def call(name: str, **args: Any) -> ToolCall:
        return ToolCall(id="rules", name=name, arguments=args)

    if REMIND.search(q):
        m = REMIND_WHAT.search(q)
        return call("reminders.add", text=(m["t"].strip() if m else q), at=q)
    if CAMERA_ON.search(q) and not CAMERA_OFF.search(q):
        return call("camera.pause", action="resume")
    if CAMERA_OFF.search(q):
        return call("camera.pause", action="pause")
    if ADD.search(q):
        m = TITLE_CALLED.search(q)
        who = WITH_WHOM.search(q)
        title = m["t"].strip() if m else (f"Meeting with {who['who']}" if who else "")
        if m and who and who["who"].lower() not in title.lower():
            title = f"{title} with {who['who']}"
        return call("calendar.add", title=title, start=q, attendees=[who["who"]] if who else [])
    if ROOM.search(q):
        return call("room.map")
    if WHO.search(q):
        return call("people.who_is_here")
    if BRAIN.search(q):
        return call("friends.shared_brain")
    if LAB.search(q):
        return call("learning.status")
    if MODEL.search(q):
        return call("box.model")
    if ENERGY.search(q) and re.search(r"\b(?:answer|reply|question|that)\b", q, re.I):
        return call("box.energy_last_answer")
    if m := RECAP.search(q):
        return call("memory.recap", part=m["part"])
    if CAMERA_Q.search(q):
        return call("camera.status")
    if CALENDAR.search(q):
        m = DAY_WORD.search(q)
        return call("calendar.list", day=m.group(0).lower() if m else "today")
    if m := OPEN.search(q):
        return call("navigate.screen", screen=m["s"])
    if INSPECTION.search(q):
        return call("inspection.status")
    return None


# --- the network brain ------------------------------------------------------------------------------------------

@dataclass
class NetworkBrain:
    """An OpenAI-style chat server reached over the network (the lab's teacher behind the hub proxy)."""

    base_url: str  # ".../v1"
    headers: dict = field(default_factory=dict)
    model: str = TEACHER_MODEL
    timeout_s: float = NETWORK_TIMEOUT_S
    transport: httpx.BaseTransport | None = None

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "NetworkBrain | None":
        """ATLAS_NETWORK_BRAIN=off turns it off. ATLAS_NETWORK_BRAIN_URL (a /v1 base) with an optional
        ATLAS_NETWORK_BRAIN_TOKEN, else the teacher pod's hub account (AUP_HUB_URL_C, AUP_HUB_USER_C, AUP_HUB_TOKEN_C):
        https://<hub>/user/<user>/proxy/8000/v1. None when neither is set (the box has no way to reach it)."""
        env = os.environ if environ is None else environ
        if str(env.get("ATLAS_NETWORK_BRAIN", "")).strip().lower() in ("0", "off", "false", "no"):
            return None
        model = env.get("ATLAS_NETWORK_BRAIN_MODEL") or TEACHER_MODEL
        if url := env.get("ATLAS_NETWORK_BRAIN_URL"):
            token = env.get("ATLAS_NETWORK_BRAIN_TOKEN")
            return cls(url.rstrip("/"), {"Authorization": f"Bearer {token}"} if token else {}, model=model)
        hub, user, token = (env.get(f"AUP_HUB_{k}_C") for k in ("URL", "USER", "TOKEN"))
        if hub and user and token:
            return cls(f"{hub.rstrip('/')}/user/{user}/proxy/{TEACHER_PORT}/v1",
                       {"Authorization": f"token {token}", "User-Agent": HUB_USER_AGENT}, model=model)
        return None

    def choose(self, messages: list[dict], tools: list[dict]) -> tuple[ToolCall | None, str]:
        """One chat call with the tools; (the tool it called or None, its text)."""
        body = {"model": self.model, "messages": messages, "tools": tools, "tool_choice": "auto", "max_tokens": 300,
                "temperature": 0.0, "chat_template_kwargs": {"enable_thinking": False}}
        with httpx.Client(timeout=self.timeout_s, transport=self.transport, follow_redirects=False) as c:
            r = c.post(f"{self.base_url}/chat/completions", json=body, headers=self.headers)
            r.raise_for_status()
            msg = r.json()["choices"][0]["message"]
        text = msg.get("content") or ""
        for c in msg.get("tool_calls") or []:
            raw = c["function"].get("arguments") or "{}"
            try:
                args = json.loads(raw) if isinstance(raw, str) else dict(raw)
            except ValueError:
                args = {}
            return ToolCall(id=c.get("id") or "net", name=c["function"]["name"], arguments=args), text
        found = calls_in_text(text)
        return (found[0] if found else None), text


# --- the router -------------------------------------------------------------------------------------------------

@dataclass
class Routed:
    text: str
    answered_by: str
    tool: str
    args: dict
    result: ToolResult
    model_version: str


class AskRouter:
    def __init__(self, runtime: Runtime, ports: Ports, clock: Callable[[], datetime],
                 publish: Callable[[Any], Awaitable[None]] | None = None, registry: ToolRegistry | None = None,
                 network: NetworkBrain | None = None, use_rules: bool = True) -> None:
        self.runtime = runtime
        self.ports = ports
        self.clock = clock
        self.publish = publish
        self.registry = registry or ToolRegistry()
        self.network = network
        self.use_rules = use_rules
        self.box_timeout_s = BOX_CHOICE_TIMEOUT_S
        self.last: Routed | None = None

    def _messages(self, question: str, now: datetime) -> list[ChatMessage]:
        local = now.astimezone(self.ports.tz)
        when = f" It is {local:%A %Y-%m-%d %H:%M} ({self.ports.tz.key})."
        return [ChatMessage(role="system", content=SYSTEM + when), ChatMessage(role="user", content=question.strip())]

    async def _box_choice(self, question: str, now: datetime) -> ToolCall | None:
        msgs = self._messages(question, now)
        try:
            r = await asyncio.wait_for(asyncio.to_thread(self.runtime.chat, msgs, self.registry.schemas()),
                                       timeout=self.box_timeout_s)
        except (TimeoutError, asyncio.TimeoutError):
            log.warning("ask router: the box's model took too long to choose a tool")
            return None
        except Exception:
            log.exception("ask router: the box's model failed to choose a tool")
            return None
        calls = [c for c in (r.tool_calls or calls_in_text(r.content)) if c.name in self.registry]
        return calls[0] if calls else None

    async def _network_choice(self, question: str, now: datetime) -> ToolCall | None:
        if self.network is None:
            return None
        msgs = [{"role": m.role, "content": m.content} for m in self._messages(question, now)]
        try:
            call, _ = await asyncio.wait_for(asyncio.to_thread(self.network.choose, msgs, self.registry.schemas()),
                                             timeout=self.network.timeout_s + 2)
        except Exception as exc:  # unreachable, a 502 from the hub, a timeout: the rules still answer
            log.warning("ask router: the network brain did not answer (%s)", type(exc).__name__)
            return None
        return call if call is not None and call.name in self.registry else None

    async def choose(self, question: str) -> tuple[ToolCall | None, str]:
        """(the tool to run, who chose it); (None, BOX) when the question is not one for the tools."""
        now = self.clock()
        call = await self._box_choice(question, now)
        if call is not None:
            return call, BOX
        if not NAMES_A_TOOL.search(question):
            return None, BOX
        call = await self._network_choice(question, now)
        if call is not None:
            return call, NETWORK
        if self.use_rules and (call := rules(question)) is not None:
            return call, RULES
        return None, BOX

    async def route(self, question: str, dry_run: bool = False) -> Routed | None:
        """The answer from a tool, or None for QA to answer as before."""
        call, by = await self.choose(question)
        if call is None:
            return None
        tool = self.registry.get(call.name)
        now = self.clock()
        result = self.registry.run(tool.name, call.arguments, self.ports, now, dry_run=dry_run)
        if not result.ok and by == BOX and (fixed := rules(question)) is not None and fixed.name == tool.name:
            # the 4B chose well but left an argument out ("add a meeting" with no title): the words have it
            merged = {**fixed.arguments, **{k: v for k, v in call.arguments.items() if v}}
            retry = self.registry.run(tool.name, merged, self.ports, now, dry_run=dry_run)
            if retry.ok:
                call, result = ToolCall(id=call.id, name=call.name, arguments=merged), retry
        if result.proposal is not None and self.publish is not None and not dry_run:
            await self.publish(result.proposal)
        version = self.runtime.model_info().version if by != NETWORK else (self.network.model if self.network else NETWORK)
        log.info("ask: answered by %s with %s %s", by, tool.name, json.dumps(call.arguments)[:200])
        routed = Routed(result.text, by, tool.name, call.arguments, result, version)
        self.last = routed
        return routed
