"""Circle: friends' Atlases plan together through Neon. Only approved summaries,
proposals, counts and scores leave the box; never frames or raw signals."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel

from contracts.events import TypedEvent, register

ShareKind = Literal["event_proposal", "reply", "habit_summary", "impact_count", "game_score"]


class Proposal(BaseModel):
    """An event one Atlas proposes to the Circle."""

    title: str
    start: datetime
    end: datetime
    location: str | None = None
    invitees: list[str] = []  # member ids


class Reply(BaseModel):
    answer: Literal["free", "busy", "counter"]
    counter: Proposal | None = None  # set when answer == "counter"


@register
class ShareQueued(TypedEvent):
    """Approved share written to the local circle_outbox; synced to Neon when online."""

    type: Literal["circle.share.queued"] = "circle.share.queued"
    share_id: str
    circle_id: str
    kind: ShareKind
    payload: dict


@register
class ShareSynced(TypedEvent):
    type: Literal["circle.share.synced"] = "circle.share.synced"
    share_id: str
    circle_id: str


@register
class MessageReceived(TypedEvent):
    """An agent_messages row from another member's Atlas."""

    type: Literal["circle.message.received"] = "circle.message.received"
    message_id: str
    circle_id: str
    thread_id: str  # the proposal this belongs to
    from_member: str
    kind: Literal["proposal", "reply"]
    proposal: Proposal | None = None
    reply: Reply | None = None


@register
class EventAgreed(TypedEvent):
    type: Literal["circle.event.agreed"] = "circle.event.agreed"
    circle_id: str
    thread_id: str
    event: Proposal
    attendees: list[str]
