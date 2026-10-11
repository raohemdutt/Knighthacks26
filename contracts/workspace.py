from datetime import datetime
from typing import Literal

from contracts.agent import ActionKind
from contracts.events import TypedEvent, register

ActionStatus = Literal["proposed", "approved", "denied", "expired", "done", "failed"]


@register
class ApprovalRequested(TypedEvent):
    type: Literal["workspace.approval.requested"] = "workspace.approval.requested"
    action_id: str
    kind: ActionKind
    target: str
    reason: str
    expires_at: datetime


@register
class FileAdded(TypedEvent):
    """A file shared from the phone into the box's inbox (W1).

    The bytes stay on the box, encrypted at rest. The event carries the extracted text (capped) so Memory's search
    index, like every projection, rebuilds from the log alone."""

    type: Literal["workspace.file.added"] = "workspace.file.added"
    file_id: str
    name: str
    content_type: str
    size: int  # bytes
    sha256: str
    text: str = ""  # extracted text, at most contexts.workspace.domain.files.TEXT_LIMIT characters; "" when unreadable
    text_chars: int = 0  # length of the full extracted text
    extractor: str | None = None  # "text", "html", "pdf", "docx"; None when nothing could be read


@register
class ActionCompleted(TypedEvent):
    """Terminal state of an action, including denied and expired ones."""

    type: Literal["workspace.action.completed"] = "workspace.action.completed"
    action_id: str
    status: ActionStatus
    result: str | None = None
