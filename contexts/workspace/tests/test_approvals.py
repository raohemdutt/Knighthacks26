from datetime import UTC, datetime, timedelta

from contexts.workspace.domain.approvals import Approvals
from contracts.agent import ActionProposed
from contracts.experience import ApprovalDecided

T0 = datetime(2026, 10, 10, 12, 0, tzinfo=UTC)


def proposed(action_id="a1"):
    return ActionProposed(
        ts=T0, action_id=action_id, kind="send_message", target="dom", reason="package", params={"text": "hi"}
    )


def decided(decision, at_s=10, action_id="a1"):
    return ApprovalDecided(ts=T0 + timedelta(seconds=at_s), action_id=action_id, decision=decision)


def test_approve_runs_the_executor_once():
    ran = []
    ap = Approvals({"send_message": lambda r: ran.append((r.target, r.params)) or "sent"})
    [req] = ap.on_proposed(proposed())
    assert req.expires_at == T0 + timedelta(minutes=5)
    [done] = ap.on_decided(decided("approve"))
    assert (done.status, done.result) == ("done", "sent")
    assert ap.on_decided(decided("approve")) == []
    assert ran == [("dom", {"text": "hi"})]


def test_deny_and_late_approve_never_run():
    ran = []
    ap = Approvals({"send_message": lambda r: ran.append(r.target) or "sent"})
    ap.on_proposed(proposed("a1"))
    ap.on_proposed(proposed("a2"))
    assert ap.on_decided(decided("deny", action_id="a1"))[0].status == "denied"
    assert ap.on_decided(decided("approve", at_s=301, action_id="a2"))[0].status == "expired"
    assert ran == []


def test_expire_sweeps_pending():
    ap = Approvals()
    ap.on_proposed(proposed())
    assert ap.expire(T0 + timedelta(minutes=4)) == []
    [gone] = ap.expire(T0 + timedelta(minutes=5))
    assert gone.status == "expired"
    assert ap.pending() == []


def test_failing_executor_reports_failed():
    def boom(request):
        raise RuntimeError("ntfy down")

    ap = Approvals({"send_message": boom})
    ap.on_proposed(proposed())
    [done] = ap.on_decided(decided("approve"))
    assert (done.status, done.result) == ("failed", "ntfy down")
