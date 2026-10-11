"""T5: the routing hook and its two policies (runtime/routing.py, contexts/telemetry/application/routing.py)."""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from contexts.telemetry.application.routing import capability, energy, policy_from, routing_snapshot
from contracts.telemetry import RequestWindow
from runtime.routing import RouteRequest, route

LABEL = RouteRequest(kind="label", source="learning", own_space=False, student_can=True)
HARD = RouteRequest(kind="label", source="learning", own_space=False, student_can=False)


def snap(student=None, teacher=None, kind="label"):
    nodes = {}
    if student is not None:
        nodes["jetson"] = {"by_kind": {kind: {"joules_per_answer": student}}}
    if teacher is not None:
        nodes["pod-a"] = {"by_kind": {kind: {"joules_per_answer": teacher}}}
    return {"nodes": nodes}


def test_capability_answers_on_the_student_when_it_can_and_escalates_otherwise():
    assert route(LABEL, None, capability).target == "student"
    d = route(HARD, None, capability)
    assert (d.target, d.policy) == ("teacher", "capability")


def test_energy_follows_only_the_snapshot_joules_per_answer():
    cheap_teacher, cheap_student = snap(student=40.0, teacher=12.0), snap(student=12.0, teacher=40.0)
    assert route(LABEL, cheap_teacher, energy).target == "teacher"  # same request, only the snapshot differs
    assert route(LABEL, cheap_student, energy).target == "student"
    assert route(LABEL, cheap_teacher, energy).policy == "energy"


def test_energy_decides_as_capability_when_the_snapshot_has_no_value_for_that_work():
    for s in (None, {"nodes": {}}, snap(student=10.0), snap(student=10.0, teacher=5.0, kind="caption")):
        assert route(LABEL, s, energy).target == "student" and route(HARD, s, energy).target == "teacher"
        assert route(LABEL, s, energy).policy == "energy>capability"


@pytest.mark.parametrize("policy", [capability, energy])
def test_the_users_own_space_never_routes_to_the_teacher_whatever_the_snapshot(policy):
    own = RouteRequest(kind="label", source="learning", own_space=True, student_can=False)
    room = RouteRequest(kind="chat", source="agent", own_space=False, student_can=False)  # not lab work
    teacher_looks_free = snap(student=1000.0, teacher=0.001, kind="label") | {}
    for request in (own, room):
        d = route(request, teacher_looks_free, policy)
        assert (d.target, d.policy) == ("student", "privacy")


def test_the_policy_comes_from_config_and_an_unknown_name_is_an_error():
    assert policy_from(None) is capability and policy_from({"routing": {"policy": "energy"}}) is energy
    with pytest.raises(ValueError, match="routing.policy"):
        policy_from({"routing": {"policy": "cheapest"}})


def test_the_choice_is_logged_on_telemetry_request_and_the_snapshot_reads_kinds():
    t0 = datetime(2026, 10, 10, 22, tzinfo=UTC)
    d = route(HARD, None, capability)
    w = RequestWindow(ts=t0, node="pod-a", request_id="r1", started=t0 - timedelta(seconds=3), ended=t0,
                      tokens_in=10, tokens_out=20, target=d.target, policy=d.policy, kind=HARD.kind)
    assert (w.target, w.policy, w.kind) == ("teacher", "capability", "label")
    values = [SimpleNamespace(node="pod-a", kind="label", joules=30.0, request_id="a", low_confidence=True),
              SimpleNamespace(node="pod-a", kind="label", joules=12.0, request_id="b", low_confidence=True),
              SimpleNamespace(node="jetson", kind=None, joules=5.0, request_id="c", low_confidence=True)]
    s = routing_snapshot(values)
    assert s["nodes"]["pod-a"]["by_kind"]["label"]["joules_per_answer"] == 12.0  # the latest
    assert "jetson" not in s["nodes"]  # a value without a kind speaks for no work
