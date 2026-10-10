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
                 clock: Callable[[], datetime], auto_reply: bool = True) -> None:
        self.member_id = member_id
        self.circle_id = circle_id
        self.store = store
        self.memory = memory
        self.clock = clock
        self.auto_reply = auto_reply  # replies carry only free/busy/counter, so they need no approval
        self._seen: datetime | None = None
        self._agreed: set[str] = set()

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
        """Run on the app's timer: surface new messages, answer proposals, detect agreement."""
        now = self.clock()
        new = sorted(self.store.since(self.circle_id, self._seen), key=lambda m: m.created_at)
        if new:
            self._seen = new[-1].created_at
        out: list = []
        for m in new:
            if m.from_member == self.member_id:
                continue
            out.append(MessageReceived(ts=now, message_id=m.id, circle_id=m.circle_id, thread_id=m.thread_id,
                                       from_member=m.from_member, kind=m.kind, proposal=m.proposal, reply=m.reply))
            invited = m.kind == "proposal" and (not m.proposal.invitees or self.member_id in m.proposal.invitees)
            if invited and self.auto_reply:
                self.store.post(AgentMessage(id=uuid4().hex, circle_id=self.circle_id, thread_id=m.thread_id,
                                             from_member=self.member_id, kind="reply", reply=self.decide(m.proposal),
                                             created_at=now))
        threads = {m.thread_id for m in new if m.kind == "reply"}
        out += [e for t in threads if (e := self._agreement(t, now))]
        return out

    # --- 4. agreement (the proposer decides) ---

    def _agreement(self, thread_id: str, now: datetime) -> EventAgreed | None:
        msgs = self.store.thread(self.circle_id, thread_id)
        proposal_msg = next((m for m in msgs if m.kind == "proposal"), None)
        if thread_id in self._agreed or proposal_msg is None or proposal_msg.from_member != self.member_id:
            return None
        invitees = set(proposal_msg.proposal.invitees)
        replies = {m.from_member: m.reply for m in msgs if m.kind == "reply"}
        if not invitees or not invitees <= replies.keys() or any(replies[i].answer != "free" for i in invitees):
            return None
        self._agreed.add(thread_id)
        self.store.record_event(self.circle_id, thread_id, proposal_msg.proposal, self.member_id)
        return EventAgreed(ts=now, circle_id=self.circle_id, thread_id=thread_id, event=proposal_msg.proposal,
                           attendees=sorted(invitees | {self.member_id}))
