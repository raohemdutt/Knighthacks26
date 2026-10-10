from typing import Literal

from contracts.events import TypedEvent, register
from contracts.types import Citation

ActionKind = Literal["export_training_data", "send_message", "browse"]


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
