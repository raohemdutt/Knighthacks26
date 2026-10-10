"""The routing hook (Context: Inference Runtime, "Routing hook"; T5): route(request, telemetry_snapshot) -> target.

    decision = route(RouteRequest(kind="label", source="learning", own_space=False, student_can=False),
                     snapshot, policy=policy_from(models_cfg))
    decision.target   # "student" (the Jetson) or "teacher" (the lab, through the hub proxy)
    decision.policy   # which policy chose, logged on telemetry.request beside `target`

The policies live in contexts/telemetry (application/routing.py), so they can change without touching runtime/.
What lives here is the one rule no policy may change: a request about the user's own space, or work the Learning
or Circle contexts do not already send to the lab, answers on the box. The hook enforces it before any policy runs,
so a snapshot that makes the teacher look cheaper can never move the user's room off the Jetson.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

Target = Literal["student", "teacher"]
LAB_SOURCES = frozenset({"learning", "circle"})  # the contexts that already send work to the lab
PRIVACY = "privacy"  # the policy name logged when the hook itself kept a request on the box


@dataclass(frozen=True)
class RouteRequest:
    kind: str  # the kind of work, as telemetry names it: "label", "caption", "chat", "eval", ...
    source: str = "agent"  # the context asking: only "learning" and "circle" may reach the lab
    own_space: bool = True  # about the user's own room, people or day: never leaves the box
    student_can: bool = True  # the on-device model can do this work


@dataclass(frozen=True)
class RouteDecision:
    target: Target
    policy: str
    reason: str


Policy = Callable[[RouteRequest, Mapping[str, Any] | None], RouteDecision]


def route(request: RouteRequest, snapshot: Mapping[str, Any] | None, policy: Policy) -> RouteDecision:
    """The target for one request. The privacy line comes first and no policy can override it."""
    if request.own_space:
        return RouteDecision("student", PRIVACY, "about the user's own space: answers on the box")
    if request.source not in LAB_SOURCES:
        return RouteDecision("student", PRIVACY, f"{request.source} work is not sent to the lab")
    return policy(request, snapshot)
