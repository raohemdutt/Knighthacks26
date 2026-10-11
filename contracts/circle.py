"""Circle: friends' Atlases plan together through Neon. Only approved summaries,
proposals, counts and scores leave the box; never frames or raw signals. A planning
question to the group brain (G4) leaves as text only, after the member's opt-in. A face template (face share,
off by default) leaves only as ciphertext sealed to one named friend's box, after an approval on both boxes."""

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
    created_by: str | None = None  # the proposer; each member's box publishes its own copy, only the proposer's sends invites


@register
class PlaceShared(TypedEvent):
    """Opt-in coarse presence (G5): a named spot, never coordinates or video, and it expires."""

    type: Literal["circle.place.shared"] = "circle.place.shared"
    circle_id: str
    member_id: str
    place: str  # one of the household's named places: "Library", "Our table"
    source: Literal["phone", "desk"]
    expires_at: datetime


# --- the group brain (G4): opt-in, text only, every send written to this box's log before it leaves ---


class BrainText(BaseModel):
    """One message sent to the group brain: a role and plain text. Never a picture, a frame or a file."""

    role: Literal["system", "user"]
    content: str


@register
class BrainConsentChanged(TypedEvent):
    """A member switched asking the group brain on or off for their own box. Nothing is sent while the member's latest
    change is off; consent_id is the opt-in reference every ask under it carries."""

    type: Literal["circle.brain.consent.changed"] = "circle.brain.consent.changed"
    consent_id: str
    circle_id: str
    member_id: str
    on: bool
    source: Literal["phone", "cli"]


@register
class BrainAsked(TypedEvent):
    """Written before a question leaves the box for the group brain: exactly the text sent, where it goes, under which
    opt-in, and the request's priority and token cap. The log is the record of everything the group brain was told."""

    type: Literal["circle.brain.asked"] = "circle.brain.asked"
    ask_id: str
    circle_id: str
    member_id: str
    opt_in_ref: str  # the consent_id of the member's circle.brain.consent.changed (on) this ask relies on
    purpose: str  # what the asker wanted, one short word: "plan", "where_to_go"
    node: str  # the pod role that answers: "pod-a", the teacher's pod
    host: str  # the hub the request goes through, e.g. "sjc.aupcloud.io"; never the account
    model: str  # the served model name asked: "group-brain", or "teacher" on a server that has only that name
    priority: int | None = None  # the vLLM request priority sent (lower is served first); None when none was sent
    max_tokens: int
    messages: list[BrainText]
    chars: int  # characters of text in messages


@register
class BrainAnswered(TypedEvent):
    """How an ask ended: the group brain's answer, or the kind of failure (the asker then falls back to its own
    model)."""

    type: Literal["circle.brain.answered"] = "circle.brain.answered"
    ask_id: str
    circle_id: str
    member_id: str
    ok: bool
    answer: str | None = None
    error: str | None = None  # when ok is false: "pod_stopped", "unreachable", "model_server_error", ...
    latency_ms: int
    model: str | None = None


# --- face share (off by default): a member's own face template, sealed to one named friend's box ---
# The sealed channel is separate from agent_messages: a box that does not know it never reads it. Version 1.

SEALED_VERSION = 1
SealedKind = Literal["box_key", "face_share", "face_revoke"]


class SealedMessage(BaseModel):
    """One row of the Circle's sealed channel. `kind` and `v` are plain values, not a closed list: a box skips
    every row whose kind or version it does not know.

    - box_key: a box's two public keys, self-signed. `to_member` is None (everyone in the Circle may read it).
    - face_share: a signed header and the ciphertext of one template, readable only by `to_member`'s box.
    - face_revoke: a signed header and nothing else; `to_member`'s box deletes what `from_member` shared.
    Nothing identity-derived is in a row except the ciphertext of a face_share."""

    id: str
    v: int = SEALED_VERSION
    circle_id: str
    from_member: str
    to_member: str | None = None
    kind: str
    body: dict
    created_at: datetime


@register
class FaceShareChanged(TypedEvent):
    """Something happened to a face share between this box's member and one friend. Ids, a state word and times
    only: never a name, a key, a vector or ciphertext.

    direction "out": this member's own template, to `peer`. shared (sent after the approval), revoked, expired.
    direction "in": `peer`'s template, to this box. offered (the card was raised), received (accepted and stored),
    declined, forgotten, revoked (by the sharer), expired, ignored (`reason` says why, in one fixed word)."""

    type: Literal["circle.face_share.changed"] = "circle.face_share.changed"
    circle_id: str
    member_id: str  # this box's member
    peer: str  # the friend's member id
    direction: Literal["out", "in"]
    state: Literal["shared", "revoked", "expired", "offered", "received", "declined", "forgotten", "ignored"]
    share_id: str | None = None
    expires_at: datetime | None = None
    reason: str | None = None  # ignored only: switched_off, unknown_member, bad_signature, replayed, expired, too_large, malformed
