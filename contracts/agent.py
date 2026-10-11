from datetime import datetime
from typing import Literal

from pydantic import BaseModel

from contracts.events import TypedEvent, register
from contracts.types import Citation

# connector_write: target is the connector tool ("calendar.create_event"), params its arguments.
# circle_share: target is the circle id, params the share payload.
ActionKind = Literal["export_training_data", "send_message", "browse", "connector_write", "circle_share",
                     "device_control"]  # device_control: the box itself (pause the cameras), after an approval


@register
class AnswerGenerated(TypedEvent):
    type: Literal["agent.answer.generated"] = "agent.answer.generated"
    question_id: str
    answer: str
    citations: list[Citation] = []
    model_version: str
    # Ask Atlas's tools (contexts/agent/application/ask_router.py), all optional: who answered ("the box", "the network
    # brain", "the box's rules"), the tool it used, the tool's result for the Ask tab, a screen of the web app to open,
    # and the approval card the tool proposed.
    answered_by: str | None = None
    tool: str | None = None
    tool_args: dict | None = None
    tool_result: dict | None = None
    link: str | None = None
    action_id: str | None = None


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


@register
class RuleAdded(TypedEvent):
    """A rule created from plain English (A3). The log is where custom rules live."""

    type: Literal["agent.rule.added"] = "agent.rule.added"
    rule: dict  # contexts.agent.domain.rules.Rule as JSON
    source_text: str


@register
class RuleToggled(TypedEvent):
    """A rule switched off or back on from the phone's Rules screen (X6). Replays, so the switch survives restarts."""

    type: Literal["agent.rule.toggled"] = "agent.rule.toggled"
    rule_id: str
    enabled: bool


@register
class NoteWritten(TypedEvent):
    """The always-on loop's hourly note (A7): a few sentences on what happened, for memory search."""

    type: Literal["agent.note.written"] = "agent.note.written"
    note_id: str
    hour: datetime  # start of the hour the note covers
    text: str
    episode_ids: list[str] = []
