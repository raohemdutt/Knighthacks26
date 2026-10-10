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
class ActionCompleted(TypedEvent):
    """Terminal state of an action, including denied and expired ones."""

    type: Literal["workspace.action.completed"] = "workspace.action.completed"
    action_id: str
    status: ActionStatus
    result: str | None = None
