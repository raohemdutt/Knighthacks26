"""A swap a gateway restart cut short after its switch reads as interrupted, not as a plain failure (additive)."""

import json

import pytest

from contexts.learning.tests.test_lab_deploy_button import box, client_for, runner_for  # noqa: F401  (box: fixture)
from runtime.modelstore import ModelStore

SWITCHED = ["STAGE verify start 1.0", "STAGE verify done 2.0", "STAGE switch start 2.0", "STAGE switch done 3.0",
            "STAGE restart start 3.0", "STAGE restart done 9.0"]
CHECKED = SWITCHED + ["STAGE check start 9.0", "STAGE check done 10.0"]


def write_status(box, *, state, returncode, lines, version="v2"):
    box["state"].mkdir(exist_ok=True)
    (box["state"] / "swap.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
    result = None if state == "running" else {"ok": state == "done", "returncode": returncode, "message": "",
                                              "rolled_back_to": None}
    (box["state"] / "status.json").write_text(json.dumps({
        "state": state, "action": "deploy", "version": version, "from": "atlas-base", "started": "x",
        "started_t": 1.0, "ended": "y", "seconds": 9.0, "pid": 999999, "result": result}), encoding="utf-8")


def deploy_view(box):
    return client_for(runner_for(box, alive=lambda pid: False)).get("/api/learning/lab/deploy").json()["deploy"]


@pytest.mark.parametrize("returncode", [-15, None])
def test_a_failed_swap_cut_short_after_its_switch_reads_interrupted(box, returncode):
    ModelStore(box["models"]).activate("v2")
    write_status(box, state="failed", returncode=returncode, lines=SWITCHED)
    view = deploy_view(box)
    assert view["state"] == "failed" and view["result"]["returncode"] == returncode   # kept for older cards
    assert view["interrupted"] is True
    assert view["message"] == "the last swap was interrupted by a restart; the box is serving v2"


def test_a_lost_running_swap_closed_from_its_log_after_the_switch_reads_interrupted(box):
    ModelStore(box["models"]).activate("v2")
    write_status(box, state="running", returncode=None, lines=SWITCHED)
    view = deploy_view(box)
    assert view["state"] == "failed" and view["interrupted"] is True and "serving v2" in view["message"]


@pytest.mark.parametrize(("serving", "returncode", "lines"), [
    ("atlas-base", -15, SWITCHED),     # the link is not the swap's version: it never switched, or went back
    ("v2", 1, SWITCHED),               # a real failure, not a restart
    ("v2", -15, CHECKED),              # the check passed: not the interrupted case
])
def test_other_failures_are_not_interrupted(box, serving, returncode, lines):
    ModelStore(box["models"]).activate(serving)
    write_status(box, state="failed", returncode=returncode, lines=lines)
    view = deploy_view(box)
    assert view["state"] == "failed" and view["interrupted"] is False and view["message"] is None


def test_idle_and_done_carry_the_fields_false(box):
    assert deploy_view(box)["interrupted"] is False
    ModelStore(box["models"]).activate("v2")
    write_status(box, state="done", returncode=0, lines=CHECKED)
    view = deploy_view(box)
    assert view["state"] == "done" and view["interrupted"] is False and view["message"] is None
