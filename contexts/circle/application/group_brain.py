"""The group brain (G4): a planning question across the Circle's members, asked of the big model on the teacher's pod.

    brain = GroupBrain("team", "ilan", TeacherLink.from_env(), bus.publish)
    bus.subscribe("circle.brain.consent", brain.on_consent, replay=True)   # state only: the opt-in, rebuilt from the log
    await brain.set_consent(True, "phone")                                 # the member's opt-in, an event in the log
    answer = await brain.ask("Where should we meet tonight?", ["Maya is at the Library until 9:00 PM"],
                             purpose="where_to_go")
    answer.text, answer.badge, answer.ask_id, answer.opt_in_ref

`link` is the way to the model (infrastructure/teacher.py: the teacher's server on the pod in the pod-a role, through the
hub proxy). The rules come from Context: Circle on Notion and the design's privacy rule (only text reaches the GPUs,
which store nothing):

- Opt-in only. Nothing is sent unless this member's latest circle.brain.consent.changed is on. The consent is an event
  in this box's log, so it outlives a restart (on_consent is replay-safe), and every ask names it as its opt_in_ref.
- Text only. A question and its facts are strings: anything else, or a data URL inside the text, is refused before
  anything is sent or logged, and the link's transport refuses the same again at the wire (lab.proxy.text_only_problem
  is the one check for both). No picture, frame or file ever leaves.
- Every send is logged first. circle.brain.asked, holding exactly the messages sent, is written to the log before the
  request leaves; if the log cannot be written, nothing is sent. circle.brain.answered follows with the answer or the
  kind of failure.
- Small and one at a time. One question is in flight per box (another is refused as busy, never queued), its text is
  capped at max_chars and its answer at the link's max_tokens, and the link asks at a planning priority that vLLM
  serves before labeling. So planning cannot crowd out the labels that share the GPU, and labeling cannot hold up a
  question behind a long run.
- Never needed. A failure is logged and raised as BrainUnavailable; the caller answers with the box's own model, and
  everything about the member's own room keeps running on the box.
"""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal, Protocol
from uuid import uuid4

from contracts.circle import BrainAnswered, BrainAsked, BrainConsentChanged, BrainText
from contracts.events import Event
from contracts.types import ChatMessage, ChatResult
from lab.proxy import text_only_problem

BADGE = "Asked the group brain"  # what the phone shows beside an answer that came from here
MAX_CHARS = 6000  # text sent per question, the instruction included
MAX_FACTS = 40
PURPOSE = re.compile(r"[a-z][a-z_]{0,31}")
SYSTEM = (
    "You are the group brain of a Circle: friends whose Atlas boxes plan together. Using only the facts given, answer "
    "the planning question in at most four short sentences. Name the times and places the facts give. Never invent a "
    "fact; if the facts do not settle it, say what is missing."
)

Reason = Literal["not_opted_in", "not_text", "too_long", "empty", "busy"]


class BrainRefused(Exception):
    """The question was refused before anything was sent or logged. reason: not_opted_in, not_text, too_long, empty
    or busy."""

    def __init__(self, reason: Reason, message: str) -> None:
        super().__init__(message)
        self.reason = reason


class BrainUnavailable(Exception):
    """The group brain gave no answer. kind is the failure's kind (pod_stopped, unreachable, model_server_error, empty,
    ...); ask_id is set when the question had already left, and then circle.brain.answered records the failure."""

    def __init__(self, kind: str, message: str, ask_id: str | None = None) -> None:
        super().__init__(message)
        self.kind = kind
        self.ask_id = ask_id


class BrainLink(Protocol):
    """The way to the model. node and host say where the text goes (a pod role, the hub); priority and max_tokens are
    what each request asks for. model() names the served model it will ask (it may ask the server, never with member
    text); chat() blocks until the answer is whole."""

    node: str
    host: str
    priority: int | None
    max_tokens: int

    def model(self) -> str: ...

    def chat(self, messages: list[ChatMessage]) -> ChatResult: ...


@dataclass(frozen=True)
class BrainAnswer:
    """The group brain's answer, with what the phone shows beside it and the log entries it can be traced to."""

    text: str
    ask_id: str
    opt_in_ref: str
    model: str
    latency_ms: int
    badge: str = BADGE


def _kind(exc: BaseException) -> str:
    kind = getattr(exc, "kind", None)
    return kind if isinstance(kind, str) and kind else type(exc).__name__


class GroupBrain:
    """One member's box asking the group brain. publish writes an event to the box's log (EventBus.publish); clock
    stamps events and timer measures the wait for an answer."""

    def __init__(self, circle_id: str, member_id: str, link: BrainLink, publish: Callable[[Event], Awaitable[None]],
                 *, clock: Callable[[], datetime] = lambda: datetime.now(UTC),
                 timer: Callable[[], float] = time.perf_counter, max_chars: int = MAX_CHARS) -> None:
        self.circle_id = circle_id
        self.member_id = member_id
        self.link = link
        self.publish = publish
        self.clock = clock
        self.timer = timer
        self.max_chars = max_chars
        self._consent: BrainConsentChanged | None = None
        self._busy = False

    # --- the opt-in -------------------------------------------------------------------------

    def on_consent(self, e: BrainConsentChanged) -> None:
        """Bus handler (replay-safe): this member's latest consent change wins."""
        if e.circle_id != self.circle_id or e.member_id != self.member_id:
            return
        if self._consent is None or e.ts >= self._consent.ts:
            self._consent = e

    @property
    def opt_in_ref(self) -> str | None:
        """The consent_id the member opted in with, or None while they have not (or have opted out)."""
        consent = self._consent
        return consent.consent_id if consent is not None and consent.on else None

    async def set_consent(self, on: bool, source: Literal["phone", "cli"]) -> BrainConsentChanged:
        """Switch asking the group brain on or off for this member: an event in the log, then in effect."""
        event = BrainConsentChanged(ts=self.clock(), consent_id=uuid4().hex, circle_id=self.circle_id,
                                    member_id=self.member_id, on=on, source=source)
        await self.publish(event)
        self.on_consent(event)
        return event

    def status(self) -> dict[str, Any]:
        """What the phone needs to show the switch: whether this member opted in, since when, and where text goes."""
        consent = self._consent
        return {"circle_id": self.circle_id, "member_id": self.member_id, "opted_in": self.opt_in_ref is not None,
                "opt_in_ref": self.opt_in_ref, "changed_at": consent.ts.isoformat() if consent is not None else None,
                "node": self.link.node, "host": self.link.host, "badge": BADGE}

    # --- asking ------------------------------------------------------------------------------

    def messages(self, question: str, facts: Sequence[str] | str = ()) -> list[ChatMessage]:
        """The messages a question and its facts become, after the text-only and size checks (BrainRefused)."""
        if isinstance(facts, str):
            facts = [facts]
        if not isinstance(question, str) or any(not isinstance(fact, str) for fact in facts):
            raise BrainRefused("not_text", "the group brain takes text only: the question and every fact are strings")
        question = " ".join(question.split())
        lines = [" ".join(fact.split()) for fact in facts if fact.strip()]
        if not question:
            raise BrainRefused("empty", "there is no question to ask")
        if len(lines) > MAX_FACTS:
            raise BrainRefused("too_long", f"{len(lines)} facts; the group brain takes at most {MAX_FACTS}")
        user = ("Facts:\n" + "\n".join(f"- {line}" for line in lines) + "\n\n" if lines else "") + f"Question: {question}"
        messages = [ChatMessage(role="system", content=SYSTEM), ChatMessage(role="user", content=user)]
        chars = sum(len(m.content) for m in messages)
        if chars > self.max_chars:
            raise BrainRefused("too_long", f"{chars} characters; the group brain takes at most {self.max_chars}")
        problem = text_only_problem({"messages": [{"role": m.role, "content": m.content} for m in messages]})
        if problem is not None:
            raise BrainRefused("not_text", f"the group brain takes text only: {problem}")
        return messages

    async def ask(self, question: str, facts: Sequence[str] | str = (), *, purpose: str = "plan") -> BrainAnswer:
        """Ask the group brain one planning question with the facts that settle it (each a line of plain text, e.g.
        a friend's shared place or reply). Refused (BrainRefused) before anything is sent or logged when the member
        has not opted in, the input is not text, it is too long or empty, or another question is in flight; on a
        failure after it left, the failure is logged and BrainUnavailable raised."""
        if not PURPOSE.fullmatch(purpose):
            raise ValueError(f"purpose is a short lowercase word like 'plan', not {purpose!r}")
        opt_in_ref = self.opt_in_ref
        if opt_in_ref is None:
            raise BrainRefused("not_opted_in", f"{self.member_id} has not opted in to asking the group brain")
        messages = self.messages(question, facts)
        if self._busy:
            raise BrainRefused("busy", "the group brain is answering this box's previous question")
        self._busy = True
        try:
            try:
                model = await asyncio.to_thread(self.link.model)
            except Exception as exc:  # nothing of the member's has left yet
                raise BrainUnavailable(_kind(exc), f"the group brain cannot be reached: {exc}") from exc
            asked = BrainAsked(
                ts=self.clock(), ask_id=uuid4().hex, circle_id=self.circle_id, member_id=self.member_id,
                opt_in_ref=opt_in_ref, purpose=purpose, node=self.link.node, host=self.link.host, model=model,
                priority=self.link.priority, max_tokens=self.link.max_tokens,
                messages=[BrainText(role=m.role, content=m.content) for m in messages],
                chars=sum(len(m.content) for m in messages))
            await self.publish(asked)  # in the log before the request leaves; if this raises, nothing is sent
            started = self.timer()
            try:
                result = await asyncio.to_thread(self.link.chat, messages)
                text = result.content.strip()
                if not text:
                    raise BrainUnavailable("empty", "the group brain answered with nothing", asked.ask_id)
            except Exception as exc:
                kind = _kind(exc)
                await self.publish(self._answered(asked, started, ok=False, error=kind))
                if isinstance(exc, BrainUnavailable):
                    raise
                raise BrainUnavailable(kind, f"the group brain did not answer: {exc}", asked.ask_id) from exc
            answered = self._answered(asked, started, ok=True, answer=text)
            await self.publish(answered)
            return BrainAnswer(text=text, ask_id=asked.ask_id, opt_in_ref=opt_in_ref, model=model,
                               latency_ms=answered.latency_ms)
        finally:
            self._busy = False

    def _answered(self, asked: BrainAsked, started: float, *, ok: bool, answer: str | None = None,
                  error: str | None = None) -> BrainAnswered:
        return BrainAnswered(ts=self.clock(), ask_id=asked.ask_id, circle_id=self.circle_id, member_id=self.member_id,
                             ok=ok, answer=answer, error=error, latency_ms=max(0, round((self.timer() - started) * 1000)),
                             model=asked.model)
