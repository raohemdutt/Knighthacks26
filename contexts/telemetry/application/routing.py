"""The two routing policies behind runtime/routing.py's hook (T5), chosen by `routing.policy` in config/models.yaml.

    policy = policy_from(models_cfg)                      # "capability" (default) or "energy"
    decision = route(request, joules.routing_snapshot(), policy)

capability: answer on the Jetson student when it can do the work, escalate to the teacher otherwise.
energy: send the work to the teacher only when the snapshot's latest joules per answer for that kind of work are lower
on the teacher than on the student; when the snapshot holds no value for that work on either node, decide as
capability does (the decision says so). Both only ever see work the hook already let through (Learning or Circle
work that is not about the user's own space).

The snapshot is the shape `routing_snapshot()` builds from the joules projection:
    {"nodes": {"jetson": {"by_kind": {"label": {"joules_per_answer": 41.0, ...}}}, "pod-a": {...}}}
"""

from collections.abc import Mapping
from typing import Any

from runtime.routing import Policy, RouteDecision, RouteRequest

STUDENT_NODE = "jetson"
TEACHER_NODE = "pod-a"  # the teacher's role (account C on the SJC hub)
CAPABILITY = "capability"
ENERGY = "energy"


def capability(request: RouteRequest, snapshot: Mapping[str, Any] | None = None) -> RouteDecision:
    if request.student_can:
        return RouteDecision("student", CAPABILITY, f"the student can do {request.kind}")
    return RouteDecision("teacher", CAPABILITY, f"the student cannot do {request.kind}")


def joules_for(snapshot: Mapping[str, Any] | None, node: str, kind: str) -> float | None:
    """The snapshot's latest joules per answer for `kind` on `node`, or None when it holds none."""
    value = (((snapshot or {}).get("nodes") or {}).get(node) or {}).get("by_kind", {}).get(kind, {}).get("joules_per_answer")
    return float(value) if isinstance(value, int | float) else None


def energy(request: RouteRequest, snapshot: Mapping[str, Any] | None) -> RouteDecision:
    student = joules_for(snapshot, STUDENT_NODE, request.kind)
    teacher = joules_for(snapshot, TEACHER_NODE, request.kind)
    if student is None or teacher is None:
        fallback = capability(request)
        return RouteDecision(fallback.target, f"{ENERGY}>{CAPABILITY}",
                             f"no joules per answer for {request.kind} on both nodes; {fallback.reason}")
    if teacher < student:
        return RouteDecision("teacher", ENERGY, f"{request.kind}: teacher {teacher:.1f} J < student {student:.1f} J per answer")
    return RouteDecision("student", ENERGY, f"{request.kind}: student {student:.1f} J <= teacher {teacher:.1f} J per answer")


POLICIES: dict[str, Policy] = {CAPABILITY: capability, ENERGY: energy}


def policy_from(models_cfg: Mapping[str, Any] | None) -> Policy:
    """`routing.policy` in config/models.yaml; capability when unset. An unknown name is an error, not a silent default."""
    name = ((models_cfg or {}).get("routing") or {}).get("policy", CAPABILITY)
    if name not in POLICIES:
        raise ValueError(f"routing.policy must be one of {sorted(POLICIES)}, not {name!r}")
    return POLICIES[name]


def routing_snapshot(values: list[Any]) -> dict[str, Any]:
    """Latest joules per answer per node and kind of work, from telemetry.joules_per_answer values (newest last).
    A value without a `kind` (older events) is skipped: it cannot speak for any kind of work."""
    nodes: dict[str, Any] = {}
    for v in values:
        kind = getattr(v, "kind", None)
        if kind is None or getattr(v, "joules", None) is None:
            continue
        nodes.setdefault(v.node, {"by_kind": {}})["by_kind"][kind] = {
            "joules_per_answer": v.joules, "request_id": v.request_id, "low_confidence": v.low_confidence}
    return {"nodes": nodes}
