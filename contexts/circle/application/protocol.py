"""Circle agent-to-agent protocol (G1).

    1. My Atlas proposes an event. My person approves the share (circle_share).
    2. The proposal is written to agent_messages in Neon.
    3. Each member's Atlas reads it, checks its OWN calendar locally, and replies
       free / busy / counter. Only that one word (and a counter time) leaves the box.
    4. When everyone invited has answered free, the proposer's Atlas publishes
       circle.event.agreed, records the event, and proposes the calendar invite.

Neon is reached through the CircleStore port. InMemoryCircleStore lets two
Atlases negotiate in one test or one laptop; the Postgres store follows
Hemdutt's N1 schema (contexts/circle/infrastructure/schema.sql).
"""

from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Protocol
from uuid import uuid4

from pydantic import BaseModel

from contexts.agent.application.memory_port import MemoryReader
from contexts.workspace.domain.approvals import ActionRequest
from contracts.agent import ActionProposed
from contracts.circle import EventAgreed, MessageReceived, Proposal, Reply

MAX_COUNTER_SHIFTS = 6  # look up to 6 hours later for a counter-proposal
# Rows are stamped by the writer's own clock. Re-read this far back each poll (message ids dedupe), so a box
# whose clock runs behind ours still gets its replies read.
CLOCK_SKEW = timedelta(minutes=10)


class AgentMessage(BaseModel):
    """One row of agent_messages."""

    id: str
    circle_id: str
    thread_id: str
    from_member: str
    kind: str  # "proposal" | "reply"
    proposal: Proposal | None = None
    reply: Reply | None = None
    created_at: datetime


class CircleStore(Protocol):
    def post(self, msg: AgentMessage) -> None: ...

    def since(self, circle_id: str, after: datetime | None) -> list[AgentMessage]: ...

    def thread(self, circle_id: str, thread_id: str) -> list[AgentMessage]: ...

    def record_event(self, circle_id: str, thread_id: str, proposal: Proposal, created_by: str) -> None: ...


class InMemoryCircleStore:
    def __init__(self) -> None:
        self.messages: list[AgentMessage] = []
        self.events: dict[str, Proposal] = {}

    def post(self, msg: AgentMessage) -> None:
        if all(m.id != msg.id for m in self.messages):
            self.messages.append(msg)

    def since(self, circle_id, after):
        return [m for m in self.messages if m.circle_id == circle_id and (after is None or m.created_at > after)]

    def thread(self, circle_id, thread_id):
        return [m for m in self.messages if m.circle_id == circle_id and m.thread_id == thread_id]

    def record_event(self, circle_id, thread_id, proposal, created_by):
        self.events.setdefault(thread_id, proposal)


def busy(memory: MemoryReader, start: datetime, end: datetime) -> bool:
    """My own calendar, checked on my own box. Events without an end count as one hour."""
    for item in memory.items(start - timedelta(hours=12), end, kind="calendar_event"):
        item_end = item.when + timedelta(hours=1)
        if item.when < end and start < item_end:
            return True
    return False


class CircleAgent:
    def __init__(self, member_id: str, circle_id: str, store: CircleStore, memory: MemoryReader,
                 clock: Callable[[], datetime], auto_reply: bool = True, members: list[str] | None = None) -> None:
        self.member_id = member_id
        self.circle_id = circle_id
        self.store = store
        self.memory = memory
        self.clock = clock
        self.auto_reply = auto_reply  # replies carry only free/busy/counter, so they need no approval
        self.members = members or []  # everyone in the circle (config); a proposal with no invitees means all of them
        self._seen: datetime | None = None
        self._handled: set[str] = set()  # message ids already acted on (rebuilt from the log on start)
        self._agreed: set[str] = set()  # threads already agreed (rebuilt from the log on start)

    # --- state from the log (replay): a restart never answers or agrees twice ---

    def on_message(self, e: MessageReceived) -> None:
        self._handled.add(e.message_id)

    def on_agreed(self, e: EventAgreed) -> None:
        self._agreed.add(e.thread_id)

    # --- 1. propose (through an approval card) ---

    def propose(self, proposal: Proposal, reason: str) -> ActionProposed:
        return ActionProposed(
            ts=self.clock(),
            action_id=uuid4().hex,
            kind="circle_share",
            target=self.circle_id,
            reason=reason,
            params={"kind": "event_proposal", "thread_id": uuid4().hex, "proposal": proposal.model_dump(mode="json")},
        )

    def share_executor(self, req: ActionRequest) -> str:
        """Workspace executor for circle_share: runs only after my person approves."""
        if req.params.get("kind") != "event_proposal":
            raise ValueError(f"unsupported share {req.params.get('kind')}")
        proposal = Proposal.model_validate(req.params["proposal"])
        self.store.post(AgentMessage(id=uuid4().hex, circle_id=self.circle_id, thread_id=req.params["thread_id"],
                                     from_member=self.member_id, kind="proposal", proposal=proposal, created_at=self.clock()))
        return f"proposed to the circle: {proposal.title}"

    # --- 2/3. read and reply ---

    def decide(self, p: Proposal) -> Reply:
        if not busy(self.memory, p.start, p.end):
            return Reply(answer="free")
        length = p.end - p.start
        for shift in range(1, MAX_COUNTER_SHIFTS + 1):
            start = p.start + timedelta(hours=shift)
            if not busy(self.memory, start, start + length):
                return Reply(answer="counter", counter=p.model_copy(update={"start": start, "end": start + length}))
        return Reply(answer="busy")

    def poll(self) -> list:
        """Run on the app's timer: surface new messages, answer proposals, follow up counters, detect agreement."""
        now = self.clock()
        after = self._seen - CLOCK_SKEW if self._seen else None
        rows = sorted(self.store.since(self.circle_id, after), key=lambda m: m.created_at)
        if rows:
            self._seen = max(self._seen or rows[-1].created_at, rows[-1].created_at)
        new = [m for m in rows if m.id not in self._handled]
        out: list = []
        for m in new:
            self._handled.add(m.id)
            if m.from_member == self.member_id:
                continue
            out.append(MessageReceived(ts=now, message_id=m.id, circle_id=m.circle_id, thread_id=m.thread_id,
                                       from_member=m.from_member, kind=m.kind, proposal=m.proposal, reply=m.reply))
            if m.kind == "proposal" and self.auto_reply and self._invited(m) and not self._replied(m.thread_id):
                self.store.post(AgentMessage(id=uuid4().hex, circle_id=self.circle_id, thread_id=m.thread_id,
                                             from_member=self.member_id, kind="reply", reply=self.decide(m.proposal),
                                             created_at=now))
            if m.kind == "reply" and m.reply.answer == "counter" and m.reply.counter:
                out += self._follow_up(m)
        threads = {m.thread_id for m in new}
        out += [e for t in threads if (e := self._agreement(t, now))]
        return out

    def _invited(self, m: AgentMessage) -> bool:
        return not m.proposal.invitees or self.member_id in m.proposal.invitees

    def _replied(self, thread_id: str) -> bool:
        """Already answered (e.g. before a restart whose log was lost): never post a second reply."""
        return any(x.kind == "reply" and x.from_member == self.member_id for x in self.store.thread(self.circle_id, thread_id))

    def _follow_up(self, m: AgentMessage) -> list[ActionProposed]:
        """A friend countered my proposal: offer my person the new time, behind an approval card like the first."""
        msgs = self.store.thread(self.circle_id, m.thread_id)
        mine = next((x for x in msgs if x.kind == "proposal"), None)
        if mine is None or mine.from_member != self.member_id:
            return []
        counter = m.reply.counter.model_copy(update={"invitees": mine.proposal.invitees})
        at = counter.start.strftime("%a %I:%M %p").replace(" 0", " ")
        return [self.propose(counter, f"{m.from_member} is busy for \"{mine.proposal.title}\" and suggested {at}: "
                                      "ask the circle about the new time?")]

    def _invitees(self, p: Proposal, proposer: str, replies: dict) -> set[str]:
        """Who must say free: the invitees; with none named, every circle member (or, without a member list, whoever replied)."""
        if p.invitees:
            return set(p.invitees)
        return (set(self.members) or set(replies)) - {proposer}

    # --- 4. agreement (the proposer decides) ---

    def _agreement(self, thread_id: str, now: datetime) -> EventAgreed | None:
        """Everyone invited said free. Every member's box sees it (so each phone hears about it); only the
        proposer records it on the shared calendar, and only the proposer's box sends invites (delivery.py)."""
        if thread_id in self._agreed:
            return None
        msgs = self.store.thread(self.circle_id, thread_id)
        proposal_msg = next((m for m in msgs if m.kind == "proposal"), None)
        if proposal_msg is None:
            return None
        proposer = proposal_msg.from_member
        replies = {m.from_member: m.reply for m in msgs if m.kind == "reply"}
        invitees = self._invitees(proposal_msg.proposal, proposer, replies)
        if self.member_id != proposer and self.member_id not in invitees:
            return None
        if not invitees or not invitees <= replies.keys() or any(replies[i].answer != "free" for i in invitees):
            return None
        self._agreed.add(thread_id)
        event = proposal_msg.proposal.model_copy(update={"invitees": sorted(invitees)})
        if self.member_id == proposer:
            self.store.record_event(self.circle_id, thread_id, event, proposer)
        return EventAgreed(ts=now, circle_id=self.circle_id, thread_id=thread_id, event=event,
                           attendees=sorted(invitees | {proposer}), created_by=proposer)
