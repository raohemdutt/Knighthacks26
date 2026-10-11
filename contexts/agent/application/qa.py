"""Question answering (A1 tools, A2 loop), on the on-device Atlas model.

A small quantized model is shaky at multi-step tool use, so common question
shapes are pre-fetched in code and the model gets one short prompt with the
evidence. Only open-ended questions go through the tool loop (max 3 calls).
No evidence means "I didn't see that." without asking the model at all. When the box
has web tools, an unsure fixed path for a question that may not be about the room
(calendar, a time window, "when was the last ...": nothing found, or the model says it
has no evidence) falls back to the tool loop instead (A13). Questions about objects in
the room ("where is my wallet?") never do: if Atlas didn't see it, it says so at once.

OWNER'S RULE (Hemdutt): Atlas talks like a person and never reads a record out. Answers written here from the data
(`Evidence.direct`: the calendar, what is due, the last visit) are worded by
contexts/day/domain/ear.py: one or two short sentences, no lists, no time ranges, no ids, tags, brackets, camera or
zone names. Do not go back to joining titles and timestamps; contexts/agent/tests/test_natural_speech.py holds it.
"""

import asyncio
import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from pathlib import Path
from time import monotonic
from zoneinfo import ZoneInfo

import yaml

from contexts.agent.application import schedule
from contexts.agent.application.digest import Digester
from contexts.agent.application.memory_port import MemoryReader
from contexts.agent.application.web_port import NOT_FOUND, SUMMARY_TITLE, WebTools
from contexts.day.domain import ear
from contracts.agent import AnswerGenerated
from contracts.experience import QuestionAsked
from contracts.types import ChatMessage, Citation, ToolCall
from runtime.interface import Runtime

NO_EVIDENCE = "I didn't see that."
MAX_TOOL_CALLS = 3
MAX_WEB_TOOL_CALLS = 4  # web_search -> read_page -> maybe one more -> answer

SYSTEM = (
    "You are Atlas, a home camera assistant. Answer in one or two short sentences "
    "using only the evidence given. Mention times. When a line is marked \"Most recent\", "
    "a question about the last time is answered from that line. If the evidence does not answer "
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
# A13: more of what the box knows, for the tool loop. Not in TOOLS, so the learning task's dataset is unchanged.
AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "item_history",
            "description": "Everything known about one object over the last hours: where it was last seen and "
                           "the moments it appeared, newest first.",
            "parameters": {
                "type": "object",
                "properties": {"label": {"type": "string"}, "hours": {"type": "number"}},
                "required": ["label"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "upcoming",
            "description": "Calendar events, Canvas assignments, campus events and group-chat plans in the next days.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}, "days": {"type": "number"}},
                "required": [],
            },
        },
    },
]
PREPARE_TOOL = {
    "type": "function",
    "function": {
        "name": "prepare_for",
        "description": "Make a short plan for an upcoming exam, deadline or event from the calendar, sleep, "
                       "group chats and the web.",
        "parameters": {"type": "object", "properties": {"event": {"type": "string"}}, "required": ["event"]},
    },
}
# A small model often refuses in its own words instead of saying NO_EVIDENCE: treat that as unsure too (A13)
REFUSAL = re.compile(
    r"\b(?:can(?:no|['\u2019]?)t|unable to|not able to) (?:browse|access|check|search|look up|see)"
    r"|\bdon['\u2019]?t have (?:access|the ability|real[- ]time|internet|any (?:information|data))"
    r"|\bi don['\u2019]?t know\b|\bas an ai\b|\bi['\u2019]?m (?:just )?an ai\b|\bno (?:internet|web) access\b",
    re.IGNORECASE)


def unsure(text: str) -> bool:
    return text.strip() == NO_EVIDENCE or bool(REFUSAL.search(text))


# Small talk saying it cannot look something up ("I can't check the weather from here"): the web tools may. Narrower
# than REFUSAL on purpose: the persona says "I'm just an AI" and "I don't know anything about that" all the time, and
# sending those to the web turned a misheard "what caused sure to my wife?" into a search result read aloud.
CANT_LOOK = re.compile(
    r"\b(?:can(?:no|['’]?)t|unable to|not able to) (?:browse|access|check|search|look up)"
    r"|\bdon['’]?t have (?:access|real[- ]time|internet)|\bno (?:internet|web) access\b",
    re.IGNORECASE)
SENTENCE_DONE = re.compile(r"[.!?]\s")


class HeldStream:
    """Passes a small-talk reply on once its first sentence shows it is not a "can't look that up" (CANT_LOOK), which the
    tool loop replaces. Until then the reply is held, so the person never hears one answer and then another. The voice
    speaks whole sentences anyway, so holding the first one costs it nothing."""

    def __init__(self, stream: Callable[[str], None] | None):
        self.stream, self.held, self.sent, self.refused = stream, "", False, False

    def feed(self, piece: str) -> None:
        if self.sent:
            self.stream(piece)
            return
        self.held += piece
        if self.refused or CANT_LOOK.search(self.held):
            self.refused = True
        elif SENTENCE_DONE.search(self.held):
            self.flush()

    def flush(self) -> None:
        if self.stream and not self.sent and self.held:
            self.stream(self.held)
        self.sent = True


# About the world outside the room, so only the web can answer: with web tools these skip small talk (A13)
OUTSIDE = re.compile(
    r"\b(?:scores?|standings|playoffs?|box score|final score|kickoff|tip-?off)\b"
    r"|\b(?:game|match|matchup) (?:against|vs\.?|versus|with)\b|\b(?:last|latest|most recent|next|upcoming) (?:game|match)\b"
    r"|\b(?:vs\.?|versus)\b|\bwho (?:won|is winning|plays)\b|\bdid (?:ucf|uf|the knights|the gators)\b",
    re.IGNORECASE)


WEB_BUDGET_S = 25.0  # a web question answers from what it has after this long (the 4B on the Jetson is slow)
WEB_CALL_MIN_S = 3.0  # but one web call always gets at least this long
WEB_CALL_MAX_S = 12.0  # and never longer than this (a page load that hangs)
MAX_PAGES = 1  # pages read per question: each one is a slow browser load and a long prompt
WEB_TOOL_NAMES = {"web_search", "reddit_search", "read_page"}
NUMBER = re.compile(r"\d+")


def grounded(answer: str, evidence: list[str]) -> bool:
    """Every number in the answer (a score, a date, a price) appears in the evidence it was written from."""
    seen = set(NUMBER.findall(" ".join(evidence)))
    return all(n in seen for n in NUMBER.findall(answer))


WEB_HINT = (" For news, sports scores, opening hours or anything outside this room, use web_search, "
            "then read_page on the best link.")
HISTORY_HOURS = 24.0
UPCOMING_DAYS = 7.0
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

# "where was the last place you saw my water bottle?" is listed first: plain "where was" would take "the last place ..." as the item
WHERE_IS = re.compile(
    r"(?:where(?: was| is|['\u2019]s)? the last place (?:you|i) (?:saw|seen|left|put)|where did (?:you|i) last (?:see|leave|put)"
    r"|where(?: did i (?:leave|put)|['\u2019]?d i (?:leave|put)| are| is|'s|\u2019s)|have you seen|(?:i )?can(?:no|['\u2019]?)t find)"
    r" (?P<item>[a-z ]+?)[?.!, ]*$"
)
# "where is the package now?" asks about "package": words around the item that are not part of its name
ITEM_LEAD = re.compile(r"^(?:my|the|our|an?)\s+")
ITEM_TAIL = re.compile(r"\s+(?:right now|now|currently|at the moment|today|again)$")
WHAT_HAPPENED = re.compile(r"what (?:happened|became) (?:to|of) (?:my |the |our )?(?P<item>[a-z ]+?)[?.!]*$")
# "did a package arrive?", "has my order been delivered?": the same trace as "what happened to the package?"
ARRIVED = re.compile(r"(?:did|has) (?:a|an|the|my|our|any) (?P<item>[a-z ]+?) (?:arrive|come|show up|get delivered|been delivered)")
# "when did someone last leave the table?", "what time did ... last ...", "when was the last time ..."
LAST_WHEN = re.compile(r"^(?:when did|when was the last|what time did .* last)|\blast time\b")
RELEVANT = 0.75  # "when did X last": keep hits scoring at least this share of the best one
NIGHT = re.compile(r"\b(?:overnight|last night|while (?:i|we) (?:was |were )?(?:asleep|sleeping|slept|were out))\b")
RECENT = re.compile(r"\b(?:recently|just now|a minute ago|a few minutes ago)\b")
SHORT_WINDOW = timedelta(hours=3)  # up to this long, each episode is its own line and citation
LAST_N = re.compile(r"(?:last|past) (?P<n>\d+|an?) ?(?P<unit>hours?|minutes?|mins?)")
# A12: "what happened while they were here?" -- the most recent visit, however long it was
VISIT = re.compile(
    r"\bwhile (?:they|the judges?|our guests?|the visitors?|he|she|someone) (?:was|were) (?:here|at (?:the|our) table|visiting)\b"
    r"|\b(?:their|the last|the latest|that|the) visit\b"
)
DIGEST = re.compile(r"\b(?:(?:daily |day'?s? )?digest|(?:recap|summary|sum up|summari[sz]e)(?: of)? (?:my |the |our )?(?:whole )?(?:day|today))\b")
VISIT_LOOKBACK = timedelta(hours=3)
VISIT_GAP = timedelta(minutes=3)  # people out of view for longer than this end one visit
VISIT_SPOKEN = 4  # moments read out in a visit recap


@dataclass
class Evidence:
    lines: list[str] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)
    system: str | None = None  # a prompt for this shape of question, instead of SYSTEM
    direct: str | None = None  # an answer written from the data itself: no model call (lists the model garbles)
    web_summary: str | None = None  # the search API's own summary of its results, when it gives one (A13)
    room_only: bool = False  # about objects or people in the room: no tool-loop fallback when nothing is found (A13)

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


def item_name(text: str) -> str:
    """The thing asked about, without "my/the/a" in front or "now / right now / at the moment" after it."""
    text = text.strip()
    while (cut := ITEM_TAIL.sub("", ITEM_LEAD.sub("", text)).strip()) != text:
        text = cut
    return text


MAX_EPISODE_LINES = 12
SYSTEM_RECAP = (
    "You are Atlas, a home camera assistant. Using only the evidence, recap what happened in time order, "
    "one short clause per hour that had something, with its time, in under 40 words. "
    f'If the evidence is empty, say "{NO_EVIDENCE}"'
)


def tidy(text: str) -> str:
    """A small model sometimes ends a real answer with the no-evidence line; keep the answer, drop the hedge."""
    text = text.strip()
    if text != NO_EVIDENCE and text.endswith(NO_EVIDENCE):
        text = text[: -len(NO_EVIDENCE)].rstrip()
    return text


TRACE_BACK = timedelta(hours=48)  # how far back "what happened to the package?" looks for tagged episodes


def named_cameras(question: str | None, cameras: set[str]) -> set[str]:
    """Cameras a question names by place: "the door" -> {"cam-door"} when that camera exists."""
    if not question:
        return set()
    words = set(re.findall(r"[a-z]+", question.lower()))
    return {c for c in cameras if c.lower().removeprefix("cam-") in words}


def time_window(q: str, now: datetime, sleep: SleepSchedule = SleepSchedule()) -> tuple[datetime, datetime] | None:
    if NIGHT.search(q):
        return sleep.last_night(now)
    if "today" in q:
        return now.replace(hour=0, minute=0, second=0, microsecond=0), now
    if "yesterday" in q:
        midnight = now.astimezone(sleep.tz).replace(hour=0, minute=0, second=0, microsecond=0)
        return midnight - timedelta(days=1), midnight
    m = LAST_N.search(q)
    if m:
        n = 1 if m["n"] in ("a", "an") else int(m["n"])
        unit = timedelta(hours=1) if m["unit"].startswith("hour") else timedelta(minutes=1)
        return now - n * unit, now
    if "last hour" in q or "past hour" in q or RECENT.search(q):
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
        self.tools = MEMORY_TOOLS + AGENT_TOOLS + ([KEYFRAME_TOOL] if load_keyframe else []) + (WEB_TOOLS if web else [])
        self.planner = None  # set by the composition root; answers "help me prep for ..." (A8)
        self.talk = None  # set by the composition root; small talk and simple questions (talk.py)
        self.names: dict[str, str] = {}  # how a course code is said: {"COP4533": "algorithms"} (agent.yaml `talk.names`)
        self.listings = schedule.Listings()  # which stored items are campus listings, not the person's own (agent.yaml `listings:`)
        self.day_plan: Callable[[], list[dict] | None] | None = None  # today's agreed plan from now (DayLoop.blocks); set by the composition root
        self.monotonic: Callable[[], float] = monotonic  # the web time budget (A13); tests replace it
        self.on_delta: Callable[[str, str], None] | None = None  # (question_id, text): streamed answer pieces (A11)

    async def on_question(self, e: QuestionAsked) -> AnswerGenerated:
        stream = (lambda text: self.on_delta(e.question_id, text)) if self.on_delta else None
        a = await self.ask(e.text, stream=stream)
        return AnswerGenerated(question_id=e.question_id, answer=a.text, citations=a.citations, model_version=a.model_version)

    async def ask(self, question: str, stream: Callable[[str], None] | None = None) -> Answer:
        # A few narrow questions about how the person looked ("when did I last look this tired?", "why do you think I'm
        # tired?", "what helped last time?", "what usually helps when I look tired?", "what makes me look tired?", and
        # "how do you know?" right after one of them) are the Day loop's look memory's, when this box keeps one: said from what
        # it remembered, ahead of the camera recap that "last" and "when did" would otherwise pull them into.
        look = getattr(self.talk, "answer_look", None)  # a stand-in conversation (tests) may not have it
        remembered = look(question, stream) if look else None
        if remembered is not None:
            return remembered
        if self.planner and self.planner.wants(question):
            plan = await self.planner.answer(question)
            return Answer(plan.text or NO_EVIDENCE, plan.citations, plan.model_version)
        evidence = self.prefetch(question)
        if evidence is None:
            if self.web and OUTSIDE.search(question):  # a score or a game: search at once, no tool-choice round
                return await self.web_first(question, stream)
            if self.talk and self.talk.wants(question):
                if not self.web:
                    return await self.talk.reply(question, stream)
                held = HeldStream(stream)
                a = await self.talk.reply(question, held.feed if stream else None)
                if not held.sent and CANT_LOOK.search(a.text):  # "I can't browse the web": let the tools try instead
                    return await self.tool_loop(question, stream)
                held.flush()  # once part of the reply was heard, it is the answer: never a second one after it
                return a
            return await self.tool_loop(question, stream)
        unsure_ok = self.web is not None and not evidence.room_only  # A13: may fall back to the tool loop
        if not evidence:
            if unsure_ok:  # the room has nothing; a connector or the web may
                return await self.tool_loop(question, stream)
            if evidence.direct and stream:  # an empty calendar, said as that ("Nothing due tomorrow.")
                stream(evidence.direct)
            return Answer(evidence.direct or NO_EVIDENCE, [], self.runtime.model_info().version)
        if evidence.direct:
            if stream:
                stream(evidence.direct)
            return Answer(evidence.direct, evidence.citations, self.runtime.model_info().version)
        if unsure_ok:  # not streamed: an unsure answer is replaced by the tool loop's
            a = await self._answer_from(question, evidence)
            if not unsure(a.text):
                if stream:
                    stream(a.text)
                return a
            return await self.tool_loop(question, stream)
        return await self._answer_from(question, evidence, stream)

    # --- tools (A1) ---

    def last_seen(self, label: str, ev: Evidence) -> None:
        label = label.strip().lower()
        head = label.split()[-1] if label.split() else label  # "water bottle": the detector's label is "bottle"
        s = None
        for name in dict.fromkeys((label, head)):
            s = self.memory.last_seen(name) or (self.memory.last_seen(name[:-1]) if name.endswith("s") else None)
            if s:
                break
        if s:
            where = f"{s.zone_id} zone on {s.camera_id}" if s.zone_id else s.camera_id
            pos = f", {s.position}" if s.position else ""
            ev.add(f"{s.label} last seen {self._t(s.ts)} in the {where}{pos}", Citation(frame_ref=s.frame_ref, ts=s.ts))

    def episodes_between(self, start: datetime, end: datetime, ev: Evidence, question: str | None = None) -> None:
        """What happened in [start, end). A short window lists each episode with its own citation; a long one
        (overnight) is grouped by hour. A question that names a camera ("the door") keeps only that camera."""
        episodes = self.memory.episodes(start, end)
        cameras = named_cameras(question, {e.camera_id for e in episodes})
        if cameras:
            episodes = [e for e in episodes if e.camera_id in cameras]
        if end - start <= SHORT_WINDOW or cameras:
            for e in sorted(episodes, key=lambda e: e.start)[:MAX_EPISODE_LINES]:
                text = e.summary or " ".join(e.captions) or ", ".join(e.labels)
                ref = e.keyframe_refs[0] if e.keyframe_refs else None
                ev.add(f"{self._t(e.start)} on {e.camera_id}: {text}", Citation(episode_id=e.id, frame_ref=ref, ts=e.start))
            return
        ev.system = SYSTEM_RECAP  # a recap names every hour that had something, not just one of them
        for b in self.memory.window(start, end).hours:
            ref = b.keyframe_refs[0] if b.keyframe_refs else None
            eid = b.episode_ids[0] if b.episode_ids else None
            ev.add(f"{self._t(b.hour)} hour: " + " ".join(b.captions), Citation(episode_id=eid, frame_ref=ref, ts=b.hour))
            ev.citations += [Citation(episode_id=other, ts=b.hour) for other in b.episode_ids[1:]]  # every episode in the hour

    def visit(self, ev: Evidence) -> None:
        """The most recent stretch with a person in view (episodes less than VISIT_GAP apart), and everything the
        cameras saw during it. No identity: episodes carry labels and captions, never names."""
        now = self.clock()
        episodes = sorted(self.memory.episodes(now - VISIT_LOOKBACK, now), key=lambda e: e.start)
        people = [e for e in episodes if "person" in e.labels]
        if not people:
            return
        start = people[-1].start
        for before, after in zip(reversed(people[:-1]), reversed(people[1:])):
            if after.start - (before.end or before.start) > VISIT_GAP:
                break
            start = before.start
        last = people[-1].end or people[-1].start
        spoken = []
        for e in [e for e in episodes if start <= e.start <= last][:MAX_EPISODE_LINES]:
            text = e.summary or " ".join(e.captions) or ", ".join(e.labels)
            ref = e.keyframe_refs[0] if e.keyframe_refs else None
            ev.add(f"{self._t(e.start)} on {e.camera_id}: {text}", Citation(episode_id=e.id, frame_ref=ref, ts=e.start))
            spoken.append(text)
        # Written from the data, no model call: on the Jetson the 4B read "while the judges were here" literally and
        # refused when the evidence said "a courier". Captions describe people, never name them.
        # Said like a person (the owner's rule): when it started, then what happened, no timestamps read out.
        tz = self.sleep.tz
        ev.direct = ear.said_visit(start.astimezone(tz), now.astimezone(tz), spoken[:VISIT_SPOKEN], len(spoken) - VISIT_SPOKEN)

    def digest(self, q: str, ev: Evidence) -> None:
        """The day's digest (A6) as the answer. "yesterday" asks for yesterday's; anything else is today's."""
        today = self.clock().astimezone(self.sleep.tz).date()
        d = Digester(self.memory, self.clock, self.sleep.tz).build(today - timedelta(days=1) if "yesterday" in q else today)
        for m in d.moments:
            ev.lines.append(f"{self._t(m.hour)} hour: {m.text}")
        ev.citations = d.citations
        ev.lines = ev.lines or [d.text]  # an empty day still answers, with "Nothing on the cameras today so far."
        ev.direct = d.text

    def trace(self, item: str, ev: Evidence) -> None:
        """Where `item` was last seen, the latest episodes the detector tagged with it, then caption matches."""
        self.last_seen(item, ev)
        now = self.clock()
        label = item[:-1] if item.endswith("s") and not self.memory.episodes(now - TRACE_BACK, now, label=item) else item
        for e in sorted(self.memory.episodes(now - TRACE_BACK, now, label=label), key=lambda e: e.start, reverse=True)[:3]:
            text = e.summary or " ".join(e.captions) or ", ".join(e.labels)
            ev.add(f"{self._t(e.start)} on {e.camera_id}: {text}",
                   Citation(episode_id=e.id, frame_ref=(e.keyframe_refs or [None])[0], ts=e.start))
        seen = {c.episode_id for c in ev.citations}
        extra = Evidence()
        self.search_memory(item, extra, newest_first=True)
        for line, c in zip(extra.lines, extra.citations):
            if c.episode_id not in seen:
                ev.add(line, c)

    def last_time(self, question: str, ev: Evidence) -> None:
        """ "When did someone last leave the table?": the best-matching camera moments, newest first, with the
        newest one labelled as such so a small model doesn't pick an earlier match (A2 on the Jetson did)."""
        hits = [h for h in self.memory.search(question, limit=10) if not h.url]
        if not hits:
            return
        best = max(h.score for h in hits)
        top = sorted((h for h in hits if h.score >= RELEVANT * best), key=lambda h: h.ts, reverse=True)[:3]
        for i, h in enumerate(top):
            which = "Most recent" if i == 0 else "Earlier"
            ev.add(f"{which}: {self._t(h.ts)} on {h.camera_id}: {h.text}",
                   Citation(episode_id=h.episode_id, frame_ref=h.frame_ref, ts=h.ts))

    def search_memory(self, q: str, ev: Evidence, newest_first: bool = False) -> None:
        hits = self.memory.search(q)
        if newest_first:  # "when did X last..." : put the latest match on top for a small model
            hits = sorted(hits, key=lambda h: h.ts, reverse=True)
        for h in hits:
            if h.url:  # a connector item (Canvas, calendar, chat), not a camera episode
                ev.add(f"{self._t(h.ts)} [{h.kind}]: {h.text}", Citation(url=h.url, title=h.title, ts=h.ts))
            else:
                ev.add(f"{self._t(h.ts)} on {h.camera_id}: {h.text}", Citation(episode_id=h.episode_id, frame_ref=h.frame_ref, ts=h.ts))

    async def get_keyframe(self, frame_ref: str, question: str | None, ev: Evidence) -> None:
        """Caption one stored (blurred) keyframe on demand, e.g. to check what was on the desk."""
        jpeg = await asyncio.to_thread(self.load_keyframe, frame_ref)
        prompt = question or "Describe what is in this picture in one sentence."
        text = await asyncio.to_thread(self.runtime.caption, jpeg, prompt)
        ev.add(f"Keyframe {frame_ref}: {text}", Citation(frame_ref=frame_ref))

    def item_history(self, label: str, hours: float, ev: Evidence) -> None:
        """What `trace` and `last_time` gather, for one object over the last `hours` (A13, Q18)."""
        label = label.strip().lower()
        self.last_seen(label, ev)
        now = self.clock()
        start = now - timedelta(hours=max(0.25, min(hours, 24 * 7)))
        name = label[:-1] if label.endswith("s") and not self.memory.episodes(start, now, label=label) else label
        seen = set()
        for e in sorted(self.memory.episodes(start, now, label=name), key=lambda e: e.start, reverse=True)[:5]:
            text = e.summary or " ".join(e.captions) or ", ".join(e.labels)
            ev.add(f"{self._t(e.start)} on {e.camera_id}: {text}",
                   Citation(episode_id=e.id, frame_ref=(e.keyframe_refs or [None])[0], ts=e.start))
            seen.add(e.id)
        for h in sorted(self.memory.search(label), key=lambda h: h.ts, reverse=True):
            if not h.url and h.episode_id not in seen and h.ts >= start:
                ev.add(f"{self._t(h.ts)} on {h.camera_id}: {h.text}",
                       Citation(episode_id=h.episode_id, frame_ref=h.frame_ref, ts=h.ts))

    def upcoming(self, query: str | None, days: float, ev: Evidence) -> None:
        """Connector items (calendar, Canvas, campus, chats) from now to `days` ahead, matching `query` if given."""
        now = self.clock()
        items = sorted(self.memory.items(now, now + timedelta(days=max(0.1, min(days, 31)))), key=lambda i: i.when)
        words = [w for w in re.findall(r"[a-z0-9]+", (query or "").lower()) if len(w) > 2]
        if words:
            items = [i for i in items if any(w in f"{i.title} {i.summary or ''}".lower() for w in words)] or items
        for i in items[:10]:
            extra = f" ({i.summary})" if i.summary else ""
            ev.add(f"{i.title}: {self._t(i.when)}{extra} [{i.connector}]", Citation(url=i.url, title=i.title, ts=i.when))

    async def prepare_for(self, event: str, ev: Evidence) -> None:
        """The exam-prep workflow (A8) as a tool: its plan text, with the plan's own citations."""
        plan = await self.planner.answer(f"help me prep for {event}")
        if plan.text:
            ev.lines.append(f"Plan for {event}: {plan.text}")
            ev.citations += plan.citations

    # --- web tools (K5, K7) ---

    async def web_search(self, query: str, ev: Evidence) -> None:
        for r in await self.web.search(query):
            if r.get("title") == SUMMARY_TITLE and r.get("summary"):
                ev.web_summary = r["summary"]
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
        """A calendar question, answered from the items and said like a person (schedule.py and the Day context's
        ear.py): never a list of titles and timestamps. The items it names are cited by link."""
        now, tz = self.clock(), self.sleep.tz
        ask = schedule.parse(q, now, tz)
        if ask is None:
            return
        text, named, found = schedule.said(ask, self.memory, now, tz, names=self.names, listings=self.listings, plan=self.plan())
        if self.talk is not None and found:  # "Want the rest?": a "yes" goes to the conversation, which then says all of it
            self.talk.rest = (schedule.replace(ask, everything=True), now) if text.endswith(ear.MORE) else None
        offer = getattr(self.talk, "offer", None)  # "Want me to plan your day?": remembered there, so "yes" starts the plan
        if offer is not None:
            text = offer(text, now)
        elif text.endswith(ear.PLAN_OFFER):  # no conversation to take the "yes": an offer that works
            text = text[: -len(ear.PLAN_OFFER)] + ear.PLAN_ELSEWHERE
        for i in named:
            extra = f" ({i.summary})" if i.summary else ""
            ev.add(f"{i.title}: {self._t(i.when)}{extra} [{i.connector}]", Citation(url=i.url, title=i.title, ts=i.when))
        if found and not ev.lines:  # answered from the plan's blocks alone: nothing to cite, but it is an answer
            ev.lines.append(text)
        ev.direct = text

    def plan(self) -> list[dict] | None:
        """Today's agreed plan through the Day context's port; None when there is none (or no port)."""
        if not self.day_plan:
            return None
        try:
            return self.day_plan()
        except Exception:  # the plan is a nicety; an answer must not fail on it
            return None

    def prefetch(self, question: str) -> Evidence | None:
        q = question.lower().strip()
        ev = Evidence()
        if m := WHERE_IS.search(q):
            ev.room_only = True
            self.last_seen(item_name(m["item"]), ev)
            return ev
        if DIGEST.search(q):  # A6: "daily digest", "recap my day": written from the data, no model call
            ev.room_only = True
            self.digest(q, ev)
            return ev
        if schedule.ITEMS_Q.search(q) or (schedule.ITEM_THING.search(q) and schedule.FUTURE.search(q)):  # A9: "what is due this week?"
            self.upcoming_items(q, ev)
            return ev
        if m := WHAT_HAPPENED.search(q) or ARRIVED.search(q):  # "what happened to the package?" / "did a package arrive?"
            ev.room_only = True
            self.trace(item_name(m["item"]), ev)
            return ev
        if VISIT.search(q):  # before the windows: "while they were here" names no window of its own
            ev.room_only = True
            self.visit(ev)
            return ev
        # Ahead of the time-window path below, which takes anything that says "today" for a camera recap (on the box
        # "what did you do for me today?" came back as 23 seconds of "Sat 8:00 PM: I saw ..."): small talk ("how are
        # you doing today?"), Atlas's own work ("what did you do for me today?") and a question about the person's day
        # ahead belong to the conversation (talk.owns). "What did you see today?", "what happened today?" and "what
        # did I do today?" are not owned and stay with the cameras.
        owns = getattr(self.talk, "owns", None)  # a stand-in conversation (tests) may not have it
        if owns is not None and owns(question):
            return None
        # The calendar, for the ways people ask by voice ("what does my day look like?", "what do I have today?",
        # "what's next?", "do I have anything this afternoon?"): the same answer in code as "what's on my calendar
        # today?", when there is something to say. With nothing found the question goes on exactly as it always did.
        # Also ahead of the time window: "what do I have today?" is the day's rundown, not what the cameras saw today.
        if schedule.parse(question, self.clock(), self.sleep.tz):
            self.upcoming_items(q, ev)
            return ev if ev else None
        if window := time_window(q, self.clock(), self.sleep):
            self.episodes_between(*window, ev, question=q)
            return ev
        if LAST_WHEN.search(q):
            self.last_time(question, ev)
            return ev
        return None

    async def _chat(self, messages: list[ChatMessage], tools: list[dict] | None = None,
                    stream: Callable[[str], None] | None = None):
        # The runtime is a blocking HTTP call; keep it off the event loop that serves the bus.
        if stream:
            return await asyncio.to_thread(lambda: self.runtime.chat(messages, tools, on_text=stream))
        return await asyncio.to_thread(self.runtime.chat, messages, tools)

    async def _answer_from(self, question: str, ev: Evidence, stream: Callable[[str], None] | None = None) -> Answer:
        prompt = "Evidence:\n" + "\n".join(f"- {line}" for line in ev.lines) + f"\n\nQuestion: {question}"
        r = await self._chat([ChatMessage(role="system", content=ev.system or SYSTEM), ChatMessage(role="user", content=prompt)],
                             stream=stream)
        return Answer(tidy(r.content) or NO_EVIDENCE, ev.citations, r.model_version)

    # --- open-ended (tool loop) ---

    async def tool_loop(self, question: str, stream: Callable[[str], None] | None = None) -> Answer:
        now = self.clock()
        hint = WEB_HINT if self.web else ""
        msgs = [
            ChatMessage(role="system", content=f"{SYSTEM} The time now is {now.isoformat()}. Use tools to find evidence.{hint}"),
            ChatMessage(role="user", content=question),
        ]
        tools = self.tools + ([PREPARE_TOOL] if self.planner else [])
        ev = Evidence()
        searched = False  # the web was searched, by the model or by us
        pages = 0
        started = self.monotonic()

        def left() -> float:
            return WEB_BUDGET_S - (self.monotonic() - started)

        for _ in range(MAX_WEB_TOOL_CALLS if self.web else MAX_TOOL_CALLS):
            if self.web and left() <= 0:
                break  # out of time (model calls count too): answer from what was found so far
            r = await self._chat(msgs, tools)
            if not r.tool_calls:
                text = (r.content.strip() if ev else NO_EVIDENCE) or NO_EVIDENCE
                if self.web and not searched and unsure(text) and left() > 0:
                    # the model gave up without trying the web: search with the question itself, answer from that
                    found = Evidence()
                    await self._web_call(ToolCall(id="auto-web", name="web_search", arguments={"query": question}),
                                         found, left())
                    if found:
                        ev.lines += found.lines
                        ev.citations += found.citations
                        return await self._finish(question, ev, stream)
                if self.web and ev:
                    return self._checked(Answer(text, ev.citations, r.model_version), ev, stream)
                if stream:
                    stream(text)
                return Answer(text, ev.citations, r.model_version)
            msgs.append(ChatMessage(role="assistant", content=r.content, tool_calls=r.tool_calls))
            for call in r.tool_calls:
                searched = searched or call.name == "web_search"
                found = Evidence()
                note = ""
                if call.name == "read_page" and pages >= MAX_PAGES:
                    note = "skipped: one page per question; answer from what you have"
                elif self.web and call.name in WEB_TOOL_NAMES:
                    pages += call.name == "read_page"
                    if not await self._web_call(call, found, left()):
                        note = f"{call.name} timed out"
                else:
                    await self._run_tool(call, found)
                ev.lines += found.lines
                ev.citations += found.citations
                result = "\n".join(found.lines) or note or "nothing found"
                msgs.append(ChatMessage(role="tool", content=result, tool_call_id=call.id))
        if not ev:
            return Answer(NO_EVIDENCE, [], self.runtime.model_info().version)
        return await self._finish(question, ev, stream)

    async def web_first(self, question: str, stream: Callable[[str], None] | None = None) -> Answer:
        """Search with the question itself and answer from the results in one model call (checked). Saves the
        small model's tool-choice round, the slowest part of a web question on the Jetson."""
        ev = Evidence()
        await self._web_call(ToolCall(id="web-first", name="web_search", arguments={"query": question}), ev,
                             WEB_BUDGET_S)
        if not ev:
            return await self.tool_loop(question, stream)
        if direct := self._summary_answer(ev):  # the search did the job: no model call (about 1 to 2 s)
            if stream:
                stream(direct.text)
            return direct
        return await self._finish(question, ev, stream)

    def _summary_answer(self, ev: Evidence) -> Answer | None:
        """The search API's summary as the answer, when it answers the question and every number in it is also in
        one of the result pages (so a summary that invented a score is not said). Two sentences at most."""
        summary = ev.web_summary
        pages = [line for line in ev.lines if not line.startswith(SUMMARY_TITLE)]
        if not summary or NOT_FOUND.search(summary) or unsure(summary) or not grounded(summary, pages):
            return None
        text = " ".join(re.split(r"(?<=[.!?])\s+", summary.strip())[:2])
        seen, cites = set(), []
        for c in ev.citations:
            if c.url and c.url not in seen:
                seen.add(c.url)
                cites.append(c)
        return Answer(text, cites[:3], self.runtime.model_info().version)

    async def _web_call(self, call: ToolCall, ev: Evidence, left: float) -> bool:
        """One web tool call inside the time budget. False when it ran out of time (nothing is added)."""
        found = Evidence()
        try:
            await asyncio.wait_for(self._run_tool(call, found), timeout=min(WEB_CALL_MAX_S, max(WEB_CALL_MIN_S, left)))
        except (TimeoutError, asyncio.TimeoutError):  # distinct classes on Python 3.10, which the Jetson runs
            return False
        ev.lines += found.lines
        ev.citations += found.citations
        ev.web_summary = ev.web_summary or found.web_summary
        return True

    async def _finish(self, question: str, ev: Evidence, stream: Callable[[str], None] | None) -> Answer:
        if not self.web:
            return await self._answer_from(question, ev, stream)
        return self._checked(await self._answer_from(question, ev), ev, stream)  # checked before it is streamed

    def _checked(self, a: Answer, ev: Evidence, stream: Callable[[str], None] | None) -> Answer:
        """A web answer whose numbers are not in what was found is not said: the links are offered instead."""
        if not grounded(a.text, ev.lines):
            links = [c for c in ev.citations if c.url][:3]
            where = "; ".join(c.title or c.url for c in links)
            text = "I couldn't confirm that from what I found." + (f" These may have it: {where}." if where else "")
            a = Answer(text, links, a.model_version)
        if stream:
            stream(a.text)
        return a

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
            elif call.name == "item_history":
                self.item_history(str(args["label"]), float(args.get("hours") or HISTORY_HOURS), ev)
            elif call.name == "upcoming":
                self.upcoming(args.get("query"), float(args.get("days") or UPCOMING_DAYS), ev)
            elif self.planner and call.name == "prepare_for":
                await self.prepare_for(str(args["event"]), ev)
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
