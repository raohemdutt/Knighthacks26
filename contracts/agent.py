from datetime import datetime
from typing import Literal

from pydantic import BaseModel

from contracts.events import TypedEvent, register
from contracts.types import Citation

# connector_write: target is the connector tool ("calendar.create_event"), params its arguments.
# circle_share: target is the circle id, params the share payload.
ActionKind = Literal["export_training_data", "send_message", "browse", "connector_write", "circle_share"]


@register
class AnswerGenerated(TypedEvent):
    type: Literal["agent.answer.generated"] = "agent.answer.generated"
    question_id: str
    answer: str
    citations: list[Citation] = []
    model_version: str


@register
class AlertRaised(TypedEvent):
    type: Literal["agent.alert.raised"] = "agent.alert.raised"
    alert_id: str
    rule_id: str
    message: str
    evidence: list[Citation] = []


@register
class ActionProposed(TypedEvent):
    type: Literal["agent.action.proposed"] = "agent.action.proposed"
    action_id: str
    kind: ActionKind
    target: str
    reason: str
    params: dict = {}


class PlanBlock(BaseModel):
    title: str  # "Study: COP 3502 recursion"
    start: datetime
    end: datetime


@register
class PlanCreated(TypedEvent):
    """A short cross-source plan (A8): exam prep tonight, or what to do next."""

    type: Literal["agent.plan.created"] = "agent.plan.created"
    plan_id: str
    subject: str  # "COP 3502 Exam 1" or the question asked
    text: str
    citations: list[Citation] = []
    blocks: list[PlanBlock] = []  # suggested calendar blocks; proposed for approval when the calendar can write
    model_version: str
