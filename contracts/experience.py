from typing import Literal

from contracts.events import TypedEvent, register


@register
class QuestionAsked(TypedEvent):
    type: Literal["experience.question.asked"] = "experience.question.asked"
    question_id: str
    text: str


@register
class ApprovalDecided(TypedEvent):
    type: Literal["experience.approval.decided"] = "experience.approval.decided"
    action_id: str
    decision: Literal["approve", "deny"]


@register
class CorrectionSubmitted(TypedEvent):
    """Thumbs up/down on an answer, caption, or example (feeds Learning)."""

    type: Literal["experience.correction.submitted"] = "experience.correction.submitted"
    subject_ref: str  # question_id, frame_ref, or episode_id
    verdict: Literal["up", "down"]
    note: str | None = None
