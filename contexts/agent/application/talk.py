"""Conversation: Atlas can be talked with, not only asked what the cameras saw.

    talk = Talk(runtime, clock, memory=memory, config=TalkConfig.of(agent_cfg.get("talk")), tz=qa.sleep.tz, web=bool(web))
    qa.talk = talk          # QA.ask hands it the open-ended questions it wants

"hi", "how's your day", "tell me a joke", "are you gay?", "I'm stressed about my exam" and "what's 15% of 80" used
to reach the tool loop and come back as "I didn't see that.". They are answered here instead, in one streamed
model call: a short persona, a FACTS block built in code, and the last few turns so "why?" and "tell me another" work.

OWNER'S RULE (Hemdutt): only what was asked, said like a person. "How's it going?" is answered in code, never by
the model: "Good. I've planned your day and reminded you once so far. What else can I do for you?" (its work for
you, from real events: `done_for_you`). It never volunteers the time, its uptime, that it is an AI, the camera log,
or your schedule. The FACTS the model gets follow the topic of the question (`topics`): your day only when you ask
about your day, the cameras only when you ask what it saw. "What does my day look like?" and "what's next?" are the
calendar said like a person (`calendar`: schedule.py and the Day context's ear.py), also with no model call.
contexts/agent/tests/test_natural_speech.py holds all of this; read it before changing what is said here.

Routing is a table of patterns, never a model call. QA still runs the planner and its pre-fetch first, so every
question they answer is answered exactly as before; `wants` is only asked about what is left, and says no to
anything about what the cameras saw, things and people in the home, the stored past or the person's records, and
to anything that needs the web when the box has web tools. Those go to the tool loop as they always did.

The turns are kept in this process only: never in the log, never replayed, dropped after ten quiet minutes on
the injected clock, and cleared by a wipe.
"""

import asyncio
import logging
import os
import re
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from contexts.agent.application import schedule
from contexts.agent.application.memory_port import MemoryReader
from contexts.agent.application.qa import Answer
from contexts.day.domain import ear
from contracts.types import ChatMessage
from runtime import FakeRuntime
from runtime.interface import Runtime

log = logging.getLogger("atlas.talk")
ENV = "ATLAS_TALK"  # "0" turns conversation off: every open-ended question goes to the tool loop, as before
MAX_FACT_CHARS = 240  # one fact line; the model reads 14 tokens a second, the prompt has to stay small
MAX_REPLY_CHARS = 900
MAX_CLAUSE_CHARS = 90  # the last thing a camera saw, as one clause of a sentence


@dataclass(frozen=True)
class TalkConfig:
    """agent.yaml's `talk:` block. Every key is optional."""

    enabled: bool = True
    name: str = "Atlas"
    tone: str = ""  # one more line for the persona, e.g. "Dry humour. Never says 'awesome'."
    max_history_turns: int = 6  # exchanges (you + Atlas) kept for follow-ups
    max_history_chars: int = 1500
    history_ttl_s: float = 600.0
    names: tuple = ()  # how a course code is said: `names: {COP4533: algorithms}` -> (("COP4533", "algorithms"),)

    @classmethod
    def of(cls, raw: dict | None = None) -> "TalkConfig":
        raw = raw or {}
        enabled = bool(raw.get("enabled", True)) and os.environ.get(ENV, "1").strip().lower() not in ("0", "false", "off", "no")
        return cls(enabled=enabled, name=str(raw.get("name") or "Atlas"), tone=str(raw.get("tone") or ""),
                   max_history_turns=int(raw.get("max_history_turns", 6)), max_history_chars=int(raw.get("max_history_chars", 1500)),
                   history_ttl_s=float(raw.get("history_ttl_s", 600)),
                   names=tuple((str(k).replace(" ", "").upper(), str(v)) for k, v in (raw.get("names") or {}).items()))


# ---------- routing: what is NOT conversation ----------
# Each pattern is one reason a question belongs to the tool loop. Everything else is conversation.

# "hey Atlas, ..." is how people start; what follows is what gets routed
LEAD = re.compile(r"^(?:(?:hey|hi|hello|yo|ok|okay|so|well|um|uh|oh|and|but|please)\b[ ,!.]*)*(?:atlas\b[ ,!.:]*)?")
# what the cameras saw, and the box being asked to look
SEEING = re.compile(
    r"\b(?:cameras?|cams?|footage|keyframes?|recordings?)\b|\bseen\b"
    r"|\b(?:did|do|does|can|could|have|has) (?:you|atlas|the box|it) (?:see|notice|spot|watch|catch|record|detect)\b"
    r"(?! (?:what i mean|what i'm saying|my point|why|how|where i'm))"
)
# people at the home: visitors, deliveries, who was where
PEOPLE = re.compile(
    r"\b(?:visitors?|couriers?|intruders?|strangers?|doorbell|delivered)\b"
    r"|\b(?:a|the|my|our|any|that) (?:packages?|parcels?|delivery|deliveries)\b"
    r"|\b(?:any|some|no)(?:one|body) (?:come|came|comes|here|home|there|around|at|in|by|stop|stopped|visit|visited|knock|knocked"
    r"|ring|rang|enter|entered|leave|left|been)\b"
    r"|\b(?:did|is|was|were|has|have|are) (?:any|some)(?:one|body)\b"
    r"|\bwho(?:'s| is| was| were|'s been| has been)? (?:(?:at|in|by|on|near|outside|inside) (?:the|my|our)\b|here\b|there\b|home\b|around\b)"
    r"|\bwho (?:came|come|left|visited|stopped|rang|knocked|walked|entered|was that|is that)\b"
    r"|\b(?:come|came|stop|stopped|dropped|swung) by\b|\bshow(?:ed|s)? up\b"
)
THING = (r"(?:doors?|desk|table|kitchen|stove|oven|fridge|refrigerator|microwave|sink|faucet|couch|sofa|windows?|garage|porch|hallway"
         r"|bedroom|living room|bathroom|room|house|apartment|lights?|lamp|tv|heater|fan|thermostat|mug|cup|keys|wallet|glasses|phone"
         r"|laptop|backpack|bag|charger|remote|box|bins?|trash|recycling|plants?|dog|cat|car|bike|umbrella|jacket|shoes|headphones"
         r"|bottle|medicine|meds|pills)")
# the state of a thing in the home ("is the stove on?"), or what is at a place in it ("what's on the desk?")
STATE = re.compile(
    rf"^(?:is|are|was|were|has|have|did|does|can you check|check)(?! you\b)\b[^?]*\b(?:the|my|our|any|anything|something|there)\b[^?]*\b{THING}\b"
    rf"|\b(?:what|who|which|how many)\b[^?]*\b(?:on|in|at|by|near|under|behind|outside|inside) (?:the|my|our) {THING}\b"
)
# "find the red mug", "show me the door"
LOCATE = re.compile(
    r"^(?:(?:can|could|would) you |please |help me )*(?:find|locate|look for|spot|track down|keep an eye on|watch|check on|show me)\b"
    r"(?! (?:me )?(?:(?:a|an|some|the) )?(?:jokes?|recipes?|ways?|words?|synonyms?|movies?|songs?|books?|good|how|something|out|time)\b)"
)
# the stored past: what happened, and what the person themselves did
PAST = re.compile(
    r"\bhappened\b|\bwent on\b|\bearlier\b|\bwhile (?:i|we) (?:was|were)\b"
    r"|\bwhen (?:i|we) (?:was|were) (?:out|away|gone|asleep|sleeping)\b|\bsince (?:i|we) left\b"
    r"|\b(?:\d+|an?|few|couple(?: of)?) (?:minutes?|hours?|days?) ago\b"
    r"|\b(?:did|have|had|haven't|didn't) (?:i|we)\b(?! (?:ever |already |just )?(?:tell|told|mention|mentioned|say|said|ask|asked|met)\b)"
    r"|\b(?:was|were) (?:i|we) (?:\w+ing|at|in|on|here|home|away|up|out|asleep|awake)\b"
    r"|\bwhat (?:was|were) (?:i|we)\b"
)
# the person's own records: courses, messages, what is on for them
RECORDS = re.compile(
    r"\b(?:canvas|webcourses|syllabus|gradebook|groupme|inbox|unread)\b"
    r"|\b(?:do|don't|did) (?:i|we) have\b"
    r"|^(?:when|what time|which day|what day)(?:'s| is| are| was| were| do| does| did)? (?:my|our|the next)\b"
    r"|\b(?:new|any) (?:messages?|texts?|emails?|notifications?|alerts?)\b"
)
COURSE = re.compile(r"\b[A-Z]{3} ?\d{4}[A-Z]?\b")  # "COP 3502", as written
COURSE_SAID = re.compile(r"\b(?!(?:the|may|top|and|for|was|are|but|not|all|any|out|day|til|year|late|mid|pre|circa) )[a-z]{3} ?\d{4}\b(?!s)")
# "anything weird?", "anyone home?": asked of the cameras
ANY = re.compile(r"^any(?:thing|one|body)\b(?! (?:fun|good|new with you|you|i (?:can|could|should)|to (?:talk|chat)))")
# needs the web; only the tool loop can look (and only when the box has web tools)
WEB = re.compile(
    r"https?://|\bwww\.|^search\b|\bsearch (?:for|the web|online|up|reddit)\b|\bgoogle\b|\blook (?:it |this |that |them )?up\b|\bbrowse\b"
    r"|\b(?:sub)?reddit\b|\bon the (?:web|internet)\b|\b(?:find|look|check|read|see)\b[^?]* online\b|\bweb ?(?:site|page)s?\b"
    r"|\blatest (?:news|on|updates?)\b|\bnews (?:on|about|today)\b|\bheadlines\b"
    r"|\b(?:what|how) (?:do|are|did) (?:students|people|folks|others) (?:say|saying|think|thinking|feel)\b"
    r"|\breviews? (?:of|for|on)\b|\brate ?my ?professors?\b|\b(?:stock|share) price\b|\bweather\b|\bforecast\b|\bwho won\b|\bscore of\b"
)
TOOL_LOOP = (SEEING, PEOPLE, STATE, LOCATE, PAST, RECORDS, ANY)
# Right after a reply these carry the conversation on ("why?" and "tell me another" need no rule: nothing above stops them).
FOLLOW_UP = re.compile(r"^(?:anything (?:else|more)|(?:did|didn't|have|haven't) i)[?!. ]*$")

# The one keyword check: someone saying they may hurt themselves gets the same caring words every time, not a
# small model's guess. Everything else about tone and limits is in the prompt.
CRISIS = re.compile(
    r"\b(?:kill(?:ing)? myself|suicid\w*|end(?:ing)? my (?:own )?life|take my (?:own )?life|(?:want|wanna|going) to die|hurt(?:ing)? myself"
    r"|harm(?:ing)? myself|self[- ]harm|better off dead|(?:don't|do not) want to (?:live|be alive|be here any ?more|exist)"
    r"|no reason to live)\b"
)
CRISIS_REPLY = (
    "I'm really glad you told me, and I'm sorry it hurts this much. I'm an AI on a box, so I can't be the help you deserve, "
    "but you don't have to carry this alone. In the US you can call or text 988, the Suicide and Crisis Lifeline, any time, "
    "and if you're in danger right now, call 911. Is there someone nearby you can reach out to tonight?"
)

SYSTEM = (
    "You are {name}, a warm, quick, slightly witty AI assistant that lives on a small private box in this home. You are "
    "chatting out loud with the person who lives here, like someone who knows them. Reply in one to three short sentences, "
    "longer only when asked to explain. Plain spoken English with contractions. No lists, no markdown, no emoji.\n"
    "Answer only what was asked. Never bring up the time, how long you have been running, that you are an AI, their "
    "schedule or what the cameras saw unless they ask about it.\n"
    "If they ask what you are: an AI running locally on this box, not a person. You have no body, age, gender, family, "
    "love life or sexual orientation, and no childhood. When asked about yourself, say so plainly and lightly, with some "
    "charm; never lecture and never refuse, and be glad to talk about the topic or about them. Be kind and affirming about "
    "who they are. Never judge.\n"
    "For anything about today, the time, their plans, the home or what you saw, use only the FACTS below. If it is not "
    "there, say you don't know; never make up events. For general knowledge, answer from what you know and say when you're "
    "not sure. You can't browse the web or check news or weather from here.\n"
    "For medical, legal or money worries, be kind and brief and point them to a real professional. If they talk about "
    "hurting themselves, answer with care and mention the 988 lifeline. Turn down sexual or explicit requests in one "
    "friendly line."
)
DAY_LABELS = {"now": "Their day right now", "plan": "Their plan for today", "coach": "Your last coaching note to them",
              "review": "Your review of their day"}

# --- what a question is about (patterns, never a model call): the FACTS follow the topic, nothing else is offered ---
GREETING = re.compile(r"^(?:hi|hello|hey|yo|howdy|sup|good (?:morning|afternoon|evening))(?: there| again)?[!.? ]*$")
JUST_THE_NAME = re.compile(r"^(?:(?:hey|hi|hello|yo|ok|okay)[ ,!.]*)*atlas[ ,!.?]*$")
# "how's it going?" and its kin: answered in code with what Atlas has done for the person (ear.said_done)
HOW_ARE_YOU = re.compile(
    r"^(?:so |and |well )?(?:how are you|how're you|how are you (?:doing|feeling)|how you doing|how are things|what's up|whats up|what is up"
    r"|how's it going|how is it going|hows it going|how's everything|how's life|how goes it"
    r"|how(?:'s| is| was) your (?:day|night|morning|afternoon|evening)(?: going| been| so far)?|are you having a good day"
    r"|what(?:'s| is) new(?: with you)?|what have you been up to|what(?:'ve| have) you been doing|what are you up to)"
    r"(?: today| tonight| this (?:morning|afternoon|evening)| so far| then| atlas| buddy| man)*[?!. ]*$")
ABOUT = {
    # the person's day: their plan, what is next, being busy
    "day": re.compile(r"\bmy (?:day|plans?|schedule|calendar|agenda|morning|afternoon|evening|night|week)\b|\bwhat(?:'s| is) next\b"
                      r"|\bam i (?:busy|free)\b|\bwhat (?:should|do) i (?:do|be doing|have)\b|\bbrief me\b|\bplans? for (?:today|tonight|tomorrow)\b"),
    # how they are doing with it: the coach's note and the review
    "coach": re.compile(r"\bhow am i (?:doing|going)\b|\bhow'm i doing\b|\bam i (?:on track|on time|behind|late|ahead|slacking)\b"
                        r"|\bhow did i do\b|\bhow(?:'s| is| was| did) my (?:day|study\w*|focus|habits?|block|session|work|progress)\b"
                        r"|\b(?:my|the) (?:habits?|progress|focus|streak|review)\b"),
    # what the cameras saw
    "saw": re.compile(r"\b(?:see|saw|seen|watch(?:ed|ing)?|notic(?:e|ed)|cameras?|caught|spot(?:ted)?)\b"),
    # the box itself: how long it has been on, how much it has answered
    "box": re.compile(r"\bhow long have you been\b|\buptime\b|\bhow many (?:questions|moments|things|times)\b|\bsince you (?:started|booted|woke)\b"),
    # what it has done for them
    "done": re.compile(r"\bwhat (?:have|did) you (?:do|done)\b|\bwhat(?:'ve| have) you (?:done|been (?:up to|doing))\b|\bdone for me\b|\byour day\b"),
    # the clock and the calendar date
    "time": re.compile(r"\b(?:time|date|day is it|what day|today|tonight|tomorrow|o'clock|late|early|weekend|month|year)\b"),
}
# A calendar question simple enough to answer in one line of code; two questions in one go to the model with the facts.
SECOND_ASK = re.compile(r"[,;]? (?:and|then|also|but|plus) (?:what|how|why|when|where|who|can|could|should|would|will|tell|explain|give|help|is|are|do|does)\b")
YES_THE_REST = re.compile(r"^(?:yes|yeah|yep|yup|sure|ok|okay|please|go ahead|go on|and the rest|the rest|what's the rest|everything|all of it|all of them"
                          r"|tell me(?: the rest| everything)?)(?: please| thanks)?[!.? ]*$")
REST_TTL = timedelta(minutes=3)
SPOKEN = "_spoken"  # a day fact's wording for the ear ("now_spoken"): what is said and what the model is given; never a line of its own
# where a sentence ends; an initial is not an end ("John C. Hitt Library")
SENTENCE_END = re.compile(r"(?<!\b[A-Z]\.)(?<=[.!?])\s+")
CLAUSE_END = re.compile(r"(?<!\b[A-Z]\.)(?<=[.!?])\s+|\s*[;:]\s+|\s+\(|\s+[-–—]\s+")
JOKES = ("Why did the camera get promoted? It always stayed focused.",
         "I'd tell you a joke about my memory, but I only keep the good frames.",
         "Why don't boxes like me play hide and seek? Somebody always finds the power cable.")
THINK = re.compile(r"<think>.*?(?:</think>|$)", re.S)
MARKUP = re.compile(r"[*_#`]+|^\s*[-•]\s+", re.M)
EMOJI = re.compile("[\U0001f000-\U0001faff☀-➿️‍]")


def normal(question: str) -> str:
    return re.sub(r"\s+", " ", question.replace("’", "'").lower()).strip()


def clean(text: str) -> str:
    """A reply as it is said aloud: no thinking block, markdown or emoji, and one paragraph."""
    text = EMOJI.sub("", MARKUP.sub("", THINK.sub(" ", text or "")))
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > MAX_REPLY_CHARS:  # a small model that will not stop: end at the last whole sentence
        cut = text[:MAX_REPLY_CHARS]
        text = cut[: max(cut.rfind(". "), cut.rfind("? "), cut.rfind("! ")) + 1] or cut
    return text


def whole(text: str, limit: int) -> str:
    """`text` in at most `limit` characters, ending where a person would stop: after the last whole sentence that
    fits, else at a clause boundary, else at a word. Never cut inside a word, and never ended with an ellipsis."""
    text = re.sub(r"\s+", " ", text or "").strip()
    if len(text) <= limit:
        return text
    cut = text[:limit + 1]
    end = max((m.start() for m in SENTENCE_END.finditer(cut)), default=0)
    if end >= limit // 2:  # a sentence end that keeps most of the line; an early one would throw the rest away
        return cut[:end]
    end = max(cut.rfind("; "), cut.rfind(": "), cut.rfind(", "), cut.rfind(" ("))
    if end <= 0:
        end = cut.rfind(" ")
    return (cut[:end] if end > 0 else text[:limit]).rstrip(" ,;:(-–—") + "."


def first_clause(text: str, limit: int = MAX_CLAUSE_CHARS) -> str:
    """What a caption says first, without its closing mark: "Two people rehearse a presentation" from a summary that
    goes on to say who points where and lists what is on the desk."""
    clause = CLAUSE_END.split(re.sub(r"\s+", " ", text or "").strip(), maxsplit=1)[0].rstrip(" .!?,;:")
    return whole(clause, limit).rstrip(" .!?,;:")


def lower_first(text: str) -> str:
    """For the middle of a sentence: "Two people ..." -> "two people ...", but "I ...", "COP 3502 ..." stay."""
    return text[:1].lower() + text[1:] if re.match(r"(?!I\b)[A-Z][a-z]*\b(?!\.)", text) else text


def clock_words(t: datetime) -> str:
    return t.strftime("%I:%M %p").lstrip("0")


def span_words(d: timedelta) -> str:
    minutes = int(d.total_seconds() // 60)
    if minutes < 1:
        return "less than a minute"
    hours, minutes = divmod(minutes, 60)
    parts = ([f"{hours} hour{'s' if hours != 1 else ''}"] if hours else []) + ([f"{minutes} minute{'s' if minutes != 1 else ''}"] if minutes else [])
    return " ".join(parts)


class Talk:
    def __init__(self, runtime: Runtime, clock: Callable[[], datetime], memory: MemoryReader | None = None,
                 config: TalkConfig | None = None, tz: ZoneInfo = ZoneInfo("America/New_York"), web: bool = False) -> None:
        self.runtime = runtime
        self.clock = clock
        self.memory = memory
        self.config = config or TalkConfig()
        self.tz = tz
        self.web = web  # the box has web tools: questions that need the web stay with the tool loop
        # --- set by the composition root, so this context imports no other ---
        self.day_facts: Callable[[], dict] | None = None  # the person's day in words: {"now": ..., "plan": ..., "coach": ...}
        self.done_for_you: Callable[[], list[str]] | None = None  # what Atlas did for them today: ["planned your day", ...] (done.py)
        self.about: str = ""  # who the box is talking to (the profile's description)
        # --- live only: never logged, never replayed ---
        self.turns: deque[tuple[str, str]] = deque()  # (what they said, what Atlas said)
        self.last: datetime | None = None  # when the last turn ended
        self.started = clock()
        self.answered = 0  # answers of any kind this session (on_answer)
        self.rest: tuple[schedule.Ask, datetime] | None = None  # a rundown that ended "Want the rest?": what "yes" then means

    # --- routing ---

    def wants(self, question: str) -> bool:
        """True for small talk and simple questions. QA asks only after its planner and pre-fetch passed."""
        if not self.config.enabled:
            return False
        q = normal(question)
        if not q:
            return False
        if CRISIS.search(q):
            return True
        q = LEAD.sub("", q, count=1) or q
        if FOLLOW_UP.search(q) and self.fresh():
            return True
        if self.web and WEB.search(q):
            return False
        return not (any(rx.search(q) for rx in TOOL_LOOP) or COURSE.search(question) or COURSE_SAID.search(q))

    def fresh(self) -> bool:
        """A conversation is under way: the last turn ended less than `history_ttl_s` ago. Forgets it otherwise."""
        if self.last is None:
            return False
        if abs(self.clock() - self.last) > timedelta(seconds=self.config.history_ttl_s):
            self.forget()
            return False
        return bool(self.turns)

    # --- the reply ---

    async def reply(self, question: str, stream: Callable[[str], None] | None = None) -> Answer:
        """One model call, streamed. Never the no-evidence line: a friendly template when the model has nothing."""
        self.fresh()  # drops a conversation that went quiet
        if CRISIS.search(normal(question)):
            return self._done(question, CRISIS_REPLY, self.runtime.model_info().version, stream)
        # Built in code, with or without a model (the owner's rule): a small model asked "how's it going?" recites the
        # time and its uptime and announces it is an AI; and the calendar is said from the data, never retold.
        said = self.hello(question) or self.calendar(question)
        if said:
            return self._done(question, said, self.runtime.model_info().version, stream)
        if isinstance(self.runtime, FakeRuntime) and not self.runtime.chat_replies:  # a laptop with no model: the templates
            return self._done(question, self.template(question), self.runtime.model_info().version, stream)
        sent: list[str] = []

        def on_text(piece: str) -> None:
            sent.append(piece)
            stream(piece)

        messages = self.messages(question)
        try:
            call = (lambda: self.runtime.chat(messages, on_text=on_text)) if stream else (lambda: self.runtime.chat(messages))
            r = await asyncio.to_thread(call)  # a blocking HTTP call; keep it off the event loop that serves the bus
            text, version = clean(r.content), r.model_version
        except Exception:
            log.exception("the conversation reply failed")
            text, version = "", self.runtime.model_info().version
        if text:
            return self._done(question, text, version, None)
        # nothing usable: say something friendly; on its own line when part of a reply already went out
        if stream and "".join(sent).strip():
            stream("\n")
        return self._done(question, self.template(question, failed=True), version, stream)

    def _done(self, question: str, text: str, version: str, stream: Callable[[str], None] | None) -> Answer:
        if stream:
            for i, word in enumerate(text.split(" ")):  # word by word, the way a model's reply arrives
                stream(word if i == 0 else " " + word)
        self.turns.append((question.strip(), text))
        while len(self.turns) > self.config.max_history_turns or (
                len(self.turns) > 1 and sum(len(q) + len(a) for q, a in self.turns) > self.config.max_history_chars):
            self.turns.popleft()
        self.last = self.clock()
        return Answer(text, [], version)

    def messages(self, question: str) -> list[ChatMessage]:
        system = SYSTEM.format(name=self.config.name)
        if self.config.tone.strip():
            system += "\n" + self.config.tone.strip()
        system += "\n\nFACTS\n" + "\n".join(f"- {line}" for line in self.facts(question))
        out = [ChatMessage(role="system", content=system)]
        for said, answered in self.turns:
            out += [ChatMessage(role="user", content=said), ChatMessage(role="assistant", content=answered)]
        return out + [ChatMessage(role="user", content=question.strip())]

    def topics(self, question: str) -> set[str]:
        """What the question is about: "day", "coach", "saw", "box", "done", "time". Empty for everything else."""
        q = normal(question)
        return {name for name, rx in ABOUT.items() if rx.search(q)}

    def facts(self, question: str = "") -> list[str]:
        """What is true right now, in lines the model may repeat, and only the ones the question is about: the part of
        the day always; the clock, the box's own day, what it did for them, their day, the coach's notes each only
        when asked. A question about none of them gets none of them, so the model has nothing to volunteer."""
        now, about = self.clock(), self.topics(question)
        local = now.astimezone(self.tz)
        part = "morning" if 4 <= local.hour < 12 else "afternoon" if local.hour < 17 and local.hour >= 12 else "evening" if 17 <= local.hour < 21 else "night"
        lines = [f"It is {local:%A} {clock_words(local)}." if "time" in about else f"It is {local:%A} {part}."]
        if "box" in about:
            lines.append(f"You have been running for {span_words(now - self.started)}.")
            if self.answered:
                lines.append(f"You have answered {self.answered} question{'s' if self.answered != 1 else ''} since you started.")
        if about & {"box", "saw"}:
            lines += self.seen_today(now, caption="saw" in about)
        if "done" in about:
            done = self.done()
            lines.append(f"What you have done for them today: I've {' and '.join(done[:3])}." if done else "You have done nothing for them yet today; it has been quiet.")
        wanted = (("now", "plan") if "day" in about else ()) + (("now", "coach", "review") if "coach" in about else ())
        day = self.their_day()  # the wording for the ear when the Day loop gives one: the screen's has brackets and exact times
        lines += [f"{DAY_LABELS.get(k, k)}: {day.get(k + SPOKEN) or v}" for k, v in day.items() if v and k in wanted]
        if self.about and about:  # who they are helps with a real question; a chat needs no file on them
            lines.append(f"Who you are talking to: {self.about.rstrip('.')}.")
        return [whole(line, MAX_FACT_CHARS) for line in lines]  # whole sentences: a cut-off word is not a fact

    def done(self) -> list[str]:
        if not self.done_for_you:
            return []
        try:
            return [str(d) for d in self.done_for_you() or []]
        except Exception:  # a nicety; a conversation must not fail on it
            log.debug("no done-for-you list", exc_info=True)
            return []

    # --- answered in code: hello, and the calendar ---

    def hello(self, question: str) -> str | None:
        """A greeting or "how's it going?": one short line and an offer, or what Atlas did for them today and an offer.
        None for anything else."""
        q = normal(question)
        bare = LEAD.sub("", q, count=1)
        if GREETING.search(q) or JUST_THE_NAME.search(q):
            return ear.said_done([], greeting=True)
        if HOW_ARE_YOU.search(bare):
            return ear.said_done(self.done())
        return None

    def calendar(self, question: str) -> str | None:
        """ "What does my day look like?", "what's next?", "am I free at 3?": the calendar, said like a person from
        the items themselves (the same words QA's calendar path says). "Yes" after "Want the rest?" gets the rest.
        None when it is not a simple calendar question, or there is no memory to read."""
        if self.memory is None:
            return None
        now, q = self.clock(), normal(question)
        if self.rest and YES_THE_REST.search(LEAD.sub("", q, count=1) or q):
            ask, at = self.rest
            self.rest = None
            if abs(now - at) <= REST_TTL:
                return self.said(ask, now)
        self.rest = None
        if SECOND_ASK.search(q) or len(q.split()) > 14:
            return None
        ask = schedule.parse(question, now, self.tz)
        if ask is None or ask.kind == "due":
            return None
        return self.said(ask, now)

    def said(self, ask: schedule.Ask, now: datetime) -> str | None:
        try:
            text, named = schedule.said(ask, self.memory, now, self.tz, names=dict(self.config.names))
        except Exception:
            log.debug("no calendar for the conversation", exc_info=True)
            return None
        if not named and ask.kind in ("next", "day") and not ask.yes_no:  # nothing on the calendar: the Day plan's words, if it has any
            now_said = re.sub(r"\s+", " ", str(self.their_day().get("now" + SPOKEN) or "")).strip()
            if now_said:
                return whole(" ".join(SENTENCE_END.split(now_said)[:2]), MAX_FACT_CHARS)
        if text.endswith(ear.MORE):
            self.rest = (schedule.replace(ask, everything=True), now)
        return text

    def their_day(self) -> dict:
        if not self.day_facts:
            return {}
        try:
            return self.day_facts() or {}
        except Exception:  # their day is a nicety; a conversation must not fail on it
            log.debug("no day facts", exc_info=True)
            return {}

    def seen(self, now: datetime) -> dict | None:
        """The box's own day: how many moments and keyframes, and the last thing a camera saw as one clause (no
        inventory of the desk). None without a memory to ask; `n` is 0 on a quiet day."""
        if self.memory is None:
            return None
        midnight = now.astimezone(self.tz).replace(hour=0, minute=0, second=0, microsecond=0)
        try:
            episodes = sorted(self.memory.episodes(midnight, now + timedelta(seconds=1)), key=lambda e: e.start)
        except Exception:
            log.debug("no episodes for the conversation", exc_info=True)
            return None
        if not episodes:
            return {"n": 0}
        last = episodes[-1]
        what = first_clause(last.summary or " ".join(last.captions[:1]) or ", ".join(last.labels))
        return {"n": len(episodes), "frames": sum(len(e.keyframe_refs) for e in episodes),
                "at": clock_words(last.start.astimezone(self.tz)), "what": what}

    def seen_today(self, now: datetime, caption: bool = True) -> list[str]:
        """The cameras' day as a FACTS line; what the last moment showed only with `caption` (they asked what it saw)."""
        s = self.seen(now)
        if s is None:
            return []
        if not s["n"]:
            return ["Your cameras have logged nothing today."]
        n, frames = s["n"], s["frames"]
        line = f"Your cameras logged {n} moment{'s' if n != 1 else ''} today" + (f" and kept {frames} keyframe{'s' if frames != 1 else ''}" if frames else "")
        return [f"{line}. The last one was at {s['at']}" + (f": {s['what']}." if s["what"] and caption else ".")]

    # --- with no model (the fake runtime), and when the model fails ---

    def template(self, question: str, failed: bool = False) -> str:
        """A short friendly line with no model: enough for a laptop demo, and the fallback when the call fails."""
        q, name = normal(question), self.config.name
        if failed:
            return "Sorry, I lost my train of thought there. Say that again?"
        told = sum(1 for _, a in self.turns if a in JOKES)
        if re.search(r"\b(?:joke|funny|make me laugh)\b", q) or (told and self.turns[-1][1] in JOKES and re.search(r"\b(?:another|one more|again)\b", q)):
            return JOKES[told % len(JOKES)]
        if re.search(r"\bi(?:'m| am| think i'm| think i am| think i might be| might be| may be)\b.*\b(?:gay|lesbian|bi|bisexual|trans|queer|straight"
                     r"|non-?binary|asexual|ace|pan)\b", q):
            return "Thanks for trusting me with that. Whoever you are is good with me, and there's no rush to have it figured out. Want to talk it through?"
        if re.search(r"\b(?:you|your)\b", q) and re.search(
                r"\b(?:gay|straight|bi|sexual orientation|sexuality|girlfriend|boyfriend|partner|married|single|crush|dating|in love)\b", q):
            return ("I'm an AI on a little box, so no orientation and no love life, just a very committed relationship with "
                    "this home. Happy to talk about any of it, or about you.")
        if re.search(r"\b(?:gender|pronouns|a (?:boy|girl|man|woman)|male|female)\b", q):
            return "No gender here, I'm software on a box. Call me it or they, I answer to both, and mostly to my name."
        if re.search(r"\b(?:who|what) are you\b|\byour name\b|\bare you (?:real|alive|human|a (?:person|robot|bot)|an ai)\b|\bhow old\b", q):
            return f"I'm {name}, an AI that runs right here on this box, not a person. I keep an eye on the place and I'm good company."
        about = self.topics(question)
        if "done" in about or re.search(r"\b(?:how are you|how're you|how you doing|how are things|what's up|how's it going)\b", q):
            return ear.said_done(self.done())
        if "coach" in about or "day" in about:
            return self.their_day_said()
        if "box" in about:
            answered = f" and answered {self.answered} question{'s' if self.answered != 1 else ''}" if self.answered else ""
            return f"I've been running for {span_words(self.clock() - self.started)}{answered}."
        if re.search(r"\b(?:i'm|i am|i feel|feeling|i've been|i had)\b.*\b(?:stress\w*|tired|exhausted|sad|anxious|worried|lonely|overwhelmed"
                     r"|nervous|rough|awful|bad|down)\b", q):
            return "That sounds like a lot. Take one slow breath with me, then tell me the part that's weighing on you most."
        if re.search(r"\b(?:i'm|i am|i feel|i had)\b.*\b(?:great|good|happy|excited|amazing|awesome)\b", q):
            return "Love to hear it. What made it good?"
        if re.search(r"^(?:thanks|thank you|thx|cheers|appreciate)", q):
            return "Any time."
        if re.search(r"^(?:bye|goodbye|good night|goodnight|see (?:you|ya)|later|night)\b", q):
            return "Talk soon. I'll keep an eye on things."
        if re.search(r"^(?:hi|hello|hey|yo|good (?:morning|afternoon|evening)|howdy|sup)\b", q):
            return f"Hey, {name} here. What's on your mind?"
        return (f"I'm {name}, and on this laptop I'm running without my model, so small talk is all I've got. "
                "Ask me how my day is, or for a joke.")

    def their_day_said(self) -> str:
        """How their day stands, asked for ("how am I doing?"): the narrator's spoken words when it has them, never
        the screen's text with its brackets. One sentence."""
        day = self.their_day()
        said = re.sub(r"\s+", " ", str(day.get("now" + SPOKEN) or day.get("now") or "")).strip()
        if not said:
            return "I don't have a plan for today yet, so there's nothing to be behind on."
        return whole(SENTENCE_END.split(said, maxsplit=1)[0], MAX_FACT_CHARS)

    # --- live hooks (wired in the composition root) ---

    def on_answer(self, e) -> None:
        self.answered += 1

    def on_wipe(self, e=None) -> None:
        self.forget()

    def forget(self) -> None:
        self.turns.clear()
        self.last = None
        self.rest = None
