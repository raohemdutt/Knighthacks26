"""Question answering (A1 tools, A2 loop), on the on-device Atlas model.

A small quantized model is shaky at multi-step tool use, so common question
shapes are pre-fetched in code and the model gets one short prompt with the
evidence. Only open-ended questions go through the tool loop (max 3 calls).
No evidence means "I didn't see that." without asking the model at all.
"""

import asyncio
import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

from contexts.agent.application.memory_port import MemoryReader
from contexts.agent.application.web_port import WebTools
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

MEMORY_TOOLS = [
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

WEB_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web. Returns titles, links and snippets.",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "reddit_search",
            "description": "What students say on Reddit. Searches one subreddit (default: the school's).",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}, "subreddit": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_page",
            "description": "Open a link from a search result and read the parts relevant to a goal.",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string"}, "goal": {"type": "string"}},
                "required": ["url"],
            },
        },
    },
]
TOOLS = MEMORY_TOOLS  # kept for the learning task: the tool set the student model is trained on
KEYFRAME_TOOL = {
    "type": "function",
    "function": {
        "name": "get_keyframe",
        "description": "Look at one stored keyframe (a frame_ref from earlier evidence) and describe what is in it.",
        "parameters": {
            "type": "object",
            "properties": {"frame_ref": {"type": "string"}, "question": {"type": "string"}},
            "required": ["frame_ref"],
        },
    },
}
PAGE_EXCERPT = 600

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


# Item questions: either clearly about the calendar/deadlines, or a thing + a future time phrase.
ITEMS_Q = re.compile(r"\b(due|deadlines?|calendar|schedule|coming up|upcoming|happening|going on|on campus)\b")
ITEM_THING = re.compile(r"\b(assignments?|homework|exams?|quiz(zes)?|classes|events?|meetings?)\b")
FUTURE = re.compile(r"\b(tonight|today|tomorrow|this week|next week|this weekend|weekend|upcoming|soon)\b")
ITEM_KINDS = {
    "assignment": re.compile(r"\b(due|deadlines?|assignments?|homework|exams?|quiz)\b"),
    "calendar_event": re.compile(r"\b(calendar|schedule|classes|meetings?|events?|happening|going on|on campus|tonight)\b"),
}


def upcoming_window(q: str, now: datetime, tz) -> tuple[datetime, datetime]:
    """Future windows for item questions: tonight, today, tomorrow, this weekend, this week (default 7 days)."""
    local = now.astimezone(tz)
    midnight = local.replace(hour=0, minute=0, second=0, microsecond=0)
    if "tonight" in q:
        return now, midnight + timedelta(days=1, hours=3)
    if "tomorrow" in q:
        return midnight + timedelta(days=1), midnight + timedelta(days=2)
    if "today" in q:
        return now, midnight + timedelta(days=1)
    if "weekend" in q:
        saturday = midnight + timedelta(days=(5 - local.weekday()) % 7)
        return max(now, saturday), saturday + timedelta(days=2)
    return now, now + timedelta(days=7)  # "this week" and the default: the next seven days


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
        web: WebTools | None = None,
        subreddit: str = "ucf",
        load_keyframe: Callable[[str], bytes] | None = None,
    ) -> None:
        self.runtime = runtime
        self.memory = memory
        self.clock = clock
        self.sleep = sleep
        self.web = web
        self.subreddit = subreddit
        self.load_keyframe = load_keyframe  # frame_ref -> blurred JPEG bytes
        self.tools = MEMORY_TOOLS + ([KEYFRAME_TOOL] if load_keyframe else []) + (WEB_TOOLS if web else [])
        self.planner = None  # set by the composition root; answers "help me prep for ..." (A8)

    async def on_question(self, e: QuestionAsked) -> AnswerGenerated:
        a = await self.ask(e.text)
        return AnswerGenerated(question_id=e.question_id, answer=a.text, citations=a.citations, model_version=a.model_version)

    async def ask(self, question: str) -> Answer:
        if self.planner and self.planner.wants(question):
            plan = await self.planner.answer(question)
            return Answer(plan.text or NO_EVIDENCE, plan.citations, plan.model_version)
        evidence = self.prefetch(question)
        if evidence is None:
            return await self.tool_loop(question)
        if not evidence:
            return Answer(NO_EVIDENCE, [], self.runtime.model_info().version)
        return await self._answer_from(question, evidence)

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

    async def get_keyframe(self, frame_ref: str, question: str | None, ev: Evidence) -> None:
        """Caption one stored (blurred) keyframe on demand, e.g. to check what was on the desk."""
        jpeg = await asyncio.to_thread(self.load_keyframe, frame_ref)
        prompt = question or "Describe what is in this picture in one sentence."
        text = await asyncio.to_thread(self.runtime.caption, jpeg, prompt)
        ev.add(f"Keyframe {frame_ref}: {text}", Citation(frame_ref=frame_ref))

    # --- web tools (K5, K7) ---

    async def web_search(self, query: str, ev: Evidence) -> None:
        for r in await self.web.search(query):
            snippet = f": {r['summary']}" if r.get("summary") else ""
            ev.add(f"{r['title']}{snippet} ({r['url']})", Citation(url=r["url"], title=r["title"]))

    async def reddit_search(self, query: str, subreddit: str | None, ev: Evidence) -> None:
        sub = (subreddit or self.subreddit).removeprefix("r/")
        for r in await self.web.reddit(sub, query):
            ev.add(f"r/{sub} post: {r['title']} ({r['url']})", Citation(url=r["url"], title=r["title"]))

    async def read_page(self, url: str, goal: str | None, ev: Evidence) -> None:
        page = await self.web.browse(url, goal)
        if page.get("blocked"):
            ev.add(f"{url} is behind a bot check; open it on your phone", Citation(url=page["url"]))
            return
        text = page.get("summary") or page.get("text", "")[:PAGE_EXCERPT]
        if text:
            ev.add(f"{page.get('title') or page['url']}: {text}", Citation(url=page["url"], title=page.get("title")))

    # --- pre-fetch (common shapes, no tool loop) ---

    def upcoming_items(self, q: str, ev: Evidence) -> None:
        """Canvas, calendar, campus and chat items in the asked window, soonest first, each cited by link."""
        start, end = upcoming_window(q, self.clock(), self.sleep.tz)
        kinds = [k for k, rx in ITEM_KINDS.items() if rx.search(q)] or [None]
        items = sorted({i.external_id: i for k in kinds for i in self.memory.items(start, end, kind=k)}.values(),
                       key=lambda i: i.when)
        for i in items[:10]:
            extra = f" ({i.summary})" if i.summary else ""
            ev.add(f"{i.title}: {self._t(i.when)}{extra} [{i.connector}]", Citation(url=i.url, title=i.title, ts=i.when))

    def prefetch(self, question: str) -> Evidence | None:
        q = question.lower().strip()
        ev = Evidence()
        if m := WHERE_IS.search(q):
            self.last_seen(m["item"], ev)
            return ev
        if ITEMS_Q.search(q) or (ITEM_THING.search(q) and FUTURE.search(q)):  # A9: "what is due this week?"
            self.upcoming_items(q, ev)
            return ev
        if window := time_window(q, self.clock(), self.sleep):
            self.episodes_between(*window, ev)
            return ev
        if q.startswith("when did") or "last time" in q:
            self.search_memory(question, ev, newest_first=True)
            return ev
        return None

    async def _chat(self, messages: list[ChatMessage], tools: list[dict] | None = None):
        # The runtime is a blocking HTTP call; keep it off the event loop that serves the bus.
        return await asyncio.to_thread(self.runtime.chat, messages, tools)

    async def _answer_from(self, question: str, ev: Evidence) -> Answer:
        prompt = "Evidence:\n" + "\n".join(f"- {line}" for line in ev.lines) + f"\n\nQuestion: {question}"
        r = await self._chat([ChatMessage(role="system", content=SYSTEM), ChatMessage(role="user", content=prompt)])
        return Answer(r.content.strip() or NO_EVIDENCE, ev.citations, r.model_version)

    # --- open-ended (tool loop) ---

    async def tool_loop(self, question: str) -> Answer:
        now = self.clock()
        msgs = [
            ChatMessage(role="system", content=f"{SYSTEM} The time now is {now.isoformat()}. Use tools to find evidence."),
            ChatMessage(role="user", content=question),
        ]
        ev = Evidence()
        for _ in range(MAX_TOOL_CALLS):
            r = await self._chat(msgs, self.tools)
            if not r.tool_calls:
                text = r.content.strip() if ev else NO_EVIDENCE
                return Answer(text or NO_EVIDENCE, ev.citations, r.model_version)
            msgs.append(ChatMessage(role="assistant", content=r.content, tool_calls=r.tool_calls))
            for call in r.tool_calls:
                found = Evidence()
                await self._run_tool(call, found)
                ev.lines += found.lines
                ev.citations += found.citations
                result = "\n".join(found.lines) or "nothing found"
                msgs.append(ChatMessage(role="tool", content=result, tool_call_id=call.id))
        if not ev:
            return Answer(NO_EVIDENCE, [], self.runtime.model_info().version)
        return await self._answer_from(question, ev)

    def _t(self, ts: datetime) -> str:
        """Times the model sees are local to the household, e.g. "Sun 7:30 AM"."""
        return ts.astimezone(self.sleep.tz).strftime("%a %I:%M %p").replace(" 0", " ")

    async def _run_tool(self, call: ToolCall, ev: Evidence) -> None:
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
            elif self.load_keyframe and call.name == "get_keyframe":
                await self.get_keyframe(str(args["frame_ref"]), args.get("question"), ev)
            elif self.web and call.name == "web_search":
                await self.web_search(str(args["query"]), ev)
            elif self.web and call.name == "reddit_search":
                await self.reddit_search(str(args["query"]), args.get("subreddit"), ev)
            elif self.web and call.name == "read_page":
                await self.read_page(str(args["url"]), args.get("goal"), ev)
        except (KeyError, ValueError, json.JSONDecodeError):
            pass  # bad arguments from a small model: the tool returns nothing found
        except Exception as exc:  # the web is flaky; say so instead of failing the answer
            ev.lines.append(f"{call.name} failed: {exc}")
