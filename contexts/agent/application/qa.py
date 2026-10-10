"""Question answering (A1 tools, A2 loop), on the on-device Atlas model.

A small quantized model is shaky at multi-step tool use, so common question
shapes are pre-fetched in code and the model gets one short prompt with the
evidence. Only open-ended questions go through the tool loop (max 3 calls).
No evidence means "I didn't see that." without asking the model at all.
"""

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

from contexts.agent.application.memory_port import MemoryReader
from contracts.agent import AnswerGenerated
from contracts.experience import QuestionAsked
from contracts.types import ChatMessage, Citation, ToolCall
from runtime.interface import Runtime

NO_EVIDENCE = "I didn't see that."
MAX_TOOL_CALLS = 3

SYSTEM = (
    "You are Atlas, a home camera assistant. Answer in one or two short sentences "
    "using only the evidence given. Mention times. If the evidence does not answer "
    f'the question, say "{NO_EVIDENCE}"'
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_memory",
            "description": "Keyword search over captions of everything the cameras saw.",
            "parameters": {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "last_seen",
            "description": "Where and when an object (keys, wallet, package) was last seen.",
            "parameters": {"type": "object", "properties": {"label": {"type": "string"}}, "required": ["label"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "episodes_between",
            "description": "What happened between two ISO 8601 times, grouped by hour.",
            "parameters": {
                "type": "object",
                "properties": {"start": {"type": "string"}, "end": {"type": "string"}},
                "required": ["start", "end"],
            },
        },
    },
]

WHERE_IS = re.compile(r"where (?:did i (?:leave|put)|are|is|'s) (?:my |the |our )?(?P<item>[a-z ]+?)[?.!]*$")
LAST_N = re.compile(r"(?:last|past) (?P<n>\d+|an?) ?(?P<unit>hours?|minutes?|mins?)")


@dataclass
class Evidence:
    lines: list[str] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)

    def add(self, line: str, citation: Citation) -> None:
        self.lines.append(line)
        self.citations.append(citation)

    def __bool__(self) -> bool:
        return bool(self.lines)


@dataclass
class Answer:
    text: str
    citations: list[Citation]
    model_version: str


@dataclass(frozen=True)
class SleepSchedule:
    """Defines "overnight". Defaults to 8 PM to 8 AM; set per household in config/agent.yaml."""

    bedtime: time = time(20)
    wake: time = time(8)
    tz: ZoneInfo = ZoneInfo("America/New_York")

    @classmethod
    def load(cls, path: str | Path = "config/agent.yaml") -> "SleepSchedule":
        raw = (yaml.safe_load(Path(path).read_text()) or {}).get("sleep", {})
        return cls(
            bedtime=time.fromisoformat(raw.get("bedtime", "20:00")),
            wake=time.fromisoformat(raw.get("wake", "08:00")),
            tz=ZoneInfo(raw.get("timezone", "America/New_York")),
        )

    def last_night(self, now: datetime) -> tuple[datetime, datetime]:
        """The most recent sleep window; ends at `now` if they are still asleep."""
        local = now.astimezone(self.tz)
        wake = datetime.combine(local.date(), self.wake, self.tz)
        start = datetime.combine(wake.date(), self.bedtime, self.tz)
        if start >= wake:
            start -= timedelta(days=1)
        return start, min(wake, local)


def time_window(q: str, now: datetime, sleep: SleepSchedule = SleepSchedule()) -> tuple[datetime, datetime] | None:
    if "overnight" in q or "last night" in q:
        return sleep.last_night(now)
    if "today" in q:
        return now.replace(hour=0, minute=0, second=0, microsecond=0), now
    m = LAST_N.search(q)
    if m:
        n = 1 if m["n"] in ("a", "an") else int(m["n"])
        unit = timedelta(hours=1) if m["unit"].startswith("hour") else timedelta(minutes=1)
        return now - n * unit, now
    if "last hour" in q or "past hour" in q:
        return now - timedelta(hours=1), now
    return None


class QA:
    def __init__(
        self,
        runtime: Runtime,
        memory: MemoryReader,
        clock: Callable[[], datetime],
        sleep: SleepSchedule = SleepSchedule(),
    ) -> None:
        self.runtime = runtime
        self.memory = memory
        self.clock = clock
        self.sleep = sleep

    def on_question(self, e: QuestionAsked) -> AnswerGenerated:
        a = self.ask(e.text)
        return AnswerGenerated(question_id=e.question_id, answer=a.text, citations=a.citations, model_version=a.model_version)

    def ask(self, question: str) -> Answer:
        evidence = self.prefetch(question)
        if evidence is None:
            return self.tool_loop(question)
        if not evidence:
            return Answer(NO_EVIDENCE, [], self.runtime.model_info().version)
        return self._answer_from(question, evidence)

    # --- tools (A1) ---

    def last_seen(self, label: str, ev: Evidence) -> None:
        label = label.strip().lower()
        s = self.memory.last_seen(label) or (self.memory.last_seen(label[:-1]) if label.endswith("s") else None)
        if s:
            where = f"{s.zone_id} zone on {s.camera_id}" if s.zone_id else s.camera_id
            pos = f", {s.position}" if s.position else ""
            ev.add(f"{s.label} last seen {self._t(s.ts)} in the {where}{pos}", Citation(frame_ref=s.frame_ref, ts=s.ts))

    def episodes_between(self, start: datetime, end: datetime, ev: Evidence) -> None:
        for b in self.memory.window(start, end).hours:
            ref = b.keyframe_refs[0] if b.keyframe_refs else None
            eid = b.episode_ids[0] if b.episode_ids else None
            ev.add(f"{self._t(b.hour)} hour: " + " ".join(b.captions), Citation(episode_id=eid, frame_ref=ref, ts=b.hour))

    def search_memory(self, q: str, ev: Evidence, newest_first: bool = False) -> None:
        hits = self.memory.search(q)
        if newest_first:  # "when did X last..." : put the latest match on top for a small model
            hits = sorted(hits, key=lambda h: h.ts, reverse=True)
        for h in hits:
            ev.add(f"{self._t(h.ts)} on {h.camera_id}: {h.text}", Citation(episode_id=h.episode_id, frame_ref=h.frame_ref, ts=h.ts))

    # --- pre-fetch (common shapes, no tool loop) ---

    def prefetch(self, question: str) -> Evidence | None:
        q = question.lower().strip()
        ev = Evidence()
        if m := WHERE_IS.search(q):
            self.last_seen(m["item"], ev)
            return ev
        if window := time_window(q, self.clock(), self.sleep):
            self.episodes_between(*window, ev)
            return ev
        if q.startswith("when did") or "last time" in q:
            self.search_memory(question, ev, newest_first=True)
            return ev
        return None

    def _answer_from(self, question: str, ev: Evidence) -> Answer:
        prompt = "Evidence:\n" + "\n".join(f"- {line}" for line in ev.lines) + f"\n\nQuestion: {question}"
        r = self.runtime.chat([ChatMessage(role="system", content=SYSTEM), ChatMessage(role="user", content=prompt)])
        return Answer(r.content.strip() or NO_EVIDENCE, ev.citations, r.model_version)

    # --- open-ended (tool loop) ---

    def tool_loop(self, question: str) -> Answer:
        now = self.clock()
        msgs = [
            ChatMessage(role="system", content=f"{SYSTEM} The time now is {now.isoformat()}. Use tools to find evidence."),
            ChatMessage(role="user", content=question),
        ]
        ev = Evidence()
        for _ in range(MAX_TOOL_CALLS):
            r = self.runtime.chat(msgs, tools=TOOLS)
            if not r.tool_calls:
                text = r.content.strip() if ev else NO_EVIDENCE
                return Answer(text or NO_EVIDENCE, ev.citations, r.model_version)
            msgs.append(ChatMessage(role="assistant", content=r.content, tool_calls=r.tool_calls))
            for call in r.tool_calls:
                found = Evidence()
                self._run_tool(call, found)
                ev.lines += found.lines
                ev.citations += found.citations
                result = "\n".join(found.lines) or "nothing found"
                msgs.append(ChatMessage(role="tool", content=result, tool_call_id=call.id))
        if not ev:
            return Answer(NO_EVIDENCE, [], self.runtime.model_info().version)
        return self._answer_from(question, ev)

    def _t(self, ts: datetime) -> str:
        """Times the model sees are local to the household, e.g. "Sun 7:30 AM"."""
        return ts.astimezone(self.sleep.tz).strftime("%a %I:%M %p").replace(" 0", " ")

    def _run_tool(self, call: ToolCall, ev: Evidence) -> None:
        args = call.arguments
        try:
            if call.name == "search_memory":
                self.search_memory(str(args["q"]), ev)
            elif call.name == "last_seen":
                self.last_seen(str(args["label"]), ev)
            elif call.name == "episodes_between":
                tz = self.clock().tzinfo
                start, end = (datetime.fromisoformat(args[k]) for k in ("start", "end"))
                self.episodes_between(start.replace(tzinfo=start.tzinfo or tz), end.replace(tzinfo=end.tzinfo or tz), ev)
        except (KeyError, ValueError, json.JSONDecodeError):
            pass  # bad arguments from a small model: the tool returns nothing found
