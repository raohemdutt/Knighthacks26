"""Approvals (W2): nothing external happens without a tap on the phone.

proposed -> approved -> done | failed
proposed -> denied | expired

Pure state machine: events in, events out. Expiry is measured on event `ts`
so a replay of the log reaches the same states. Executors are injected per
action kind and only ever run after an approve.
"""

from collections.abc import Callable
from datetime import datetime, timedelta

from pydantic import BaseModel

from contracts.agent import ActionKind, ActionProposed
from contracts.experience import ApprovalDecided
from contracts.workspace import ActionCompleted, ActionStatus, ApprovalRequested

APPROVAL_TTL = timedelta(minutes=5)


class ActionRequest(BaseModel):
    id: str
    kind: ActionKind
    target: str
    reason: str
    params: dict = {}
    status: ActionStatus = "proposed"
    expires_at: datetime
    result: str | None = None


Executor = Callable[[ActionRequest], str]  # returns result text


class Approvals:
    def __init__(self, executors: dict[str, Executor] | None = None) -> None:
        self.executors = executors or {}
        self.requests: dict[str, ActionRequest] = {}

    def pending(self) -> list[ActionRequest]:
        return [r for r in self.requests.values() if r.status == "proposed"]

    def on_proposed(self, e: ActionProposed) -> list[ApprovalRequested]:
        if e.action_id in self.requests:
            return []
        req = ActionRequest(
            id=e.action_id,
            kind=e.kind,
            target=e.target,
            reason=e.reason,
            params=e.params,
            expires_at=e.ts + APPROVAL_TTL,
        )
        self.requests[req.id] = req
        return [
            ApprovalRequested(
                ts=e.ts,
                action_id=req.id,
                kind=req.kind,
                target=req.target,
                reason=req.reason,
                expires_at=req.expires_at,
            )
        ]

    def on_decided(self, e: ApprovalDecided) -> list[ActionCompleted]:
        req = self.requests.get(e.action_id)
        if req is None or req.status != "proposed":
            return []
        if e.ts >= req.expires_at:
            return [self._finish(req, "expired", e.ts)]
        if e.decision == "deny":
            return [self._finish(req, "denied", e.ts)]
        req.status = "approved"
        executor = self.executors.get(req.kind)
        if executor is None:
            return [self._finish(req, "failed", e.ts, f"no executor for {req.kind}")]
        try:
            result = executor(req)
        except Exception as exc:  # an action failing must not take the box down
            return [self._finish(req, "failed", e.ts, str(exc))]
        return [self._finish(req, "done", e.ts, result)]

    def expire(self, now: datetime) -> list[ActionCompleted]:
        """Call on a timer (or any event) with the current event time."""
        return [self._finish(r, "expired", now) for r in self.pending() if now >= r.expires_at]

    def _finish(self, req: ActionRequest, status: ActionStatus, ts: datetime, result: str | None = None):
        req.status = status
        req.result = result
        return ActionCompleted(ts=ts, action_id=req.id, status=status, result=result)
