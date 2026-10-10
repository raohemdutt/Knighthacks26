"""python -m contexts.circle.brain: the real command, GroupBrain and TeacherLink, with the fake teacher behind the fake
hub proxy (lab/teacher_fake.py) and the events written to a temporary folder. No .env file is read."""

import io
import json

from bus import EventBus
from contexts.circle import brain
from contexts.circle.tests.brain_helpers import ENV, FACTS, PLAN, QUESTION, TOKEN, USER, World


def cli(world, *argv, environ=ENV):
    out = io.StringIO()
    code = brain.main(list(argv), environ=environ, env_files=[], transport=world.fake.transport, stdout=out)
    text = out.getvalue()
    assert TOKEN not in text and USER not in text
    return code, json.loads(text)


def ask_args(log_dir, *extra):
    facts = [arg for fact in FACTS for arg in ("--fact", fact)]
    return ["ask", "--log-dir", str(log_dir), "--circle", "team", "--member", "ilan", *facts, *extra, QUESTION]


def logged(log_dir):
    return list(EventBus(log_dir).read_log())


def test_status_names_the_served_model_a_question_would_use_and_sends_no_text(tmp_path):
    both, only_teacher = World(tmp_path / "a"), World(tmp_path / "b", aliases=())

    assert cli(both, "status") == (0, {"ok": True, "model": "group-brain", "node": "pod-a", "host": "teacher.test",
                                       "priority": -10, "max_tokens": 256})
    assert cli(only_teacher, "status")[1]["model"] == "teacher"
    assert [seen.method for seen in both.fake.requests + only_teacher.fake.requests] == ["GET", "GET"]


def test_a_member_who_has_not_opted_in_is_refused_and_nothing_is_sent(tmp_path):
    w = World(tmp_path)

    code, report = cli(w, *ask_args(tmp_path / "log"))

    assert (code, report["ok"], report["kind"]) == (1, False, "not_opted_in")
    assert w.fake.requests == [] and logged(tmp_path / "log") == []


def test_opt_in_is_recorded_once_then_every_question_is_logged_before_it_leaves(tmp_path):
    w = World(tmp_path)
    log_dir = tmp_path / "log"

    code, first = cli(w, *ask_args(log_dir, "--opt-in", "--purpose", "where_to_go"))
    again, second = cli(w, *ask_args(log_dir, "--opt-in"))

    events = logged(log_dir)
    assert [e.type for e in events] == ["circle.brain.consent.changed", "circle.brain.asked", "circle.brain.answered",
                                        "circle.brain.asked", "circle.brain.answered"]
    consent, asked = events[0], events[1]
    assert (consent.source, consent.on, consent.member_id, consent.circle_id) == ("cli", True, "ilan", "team")
    assert (code, again) == (0, 0)
    assert first == {"ok": True, "answer": PLAN, "badge": "Asked the group brain", "ask_id": asked.ask_id,
                     "opt_in_ref": consent.consent_id, "model": "group-brain", "latency_ms": events[2].latency_ms,
                     "log_dir": str(log_dir)}
    assert second["opt_in_ref"] == consent.consent_id
    assert asked.purpose == "where_to_go"
    assert [chat.body["messages"] for chat in w.chats()] == [
        [m.model_dump() for m in e.messages] for e in events if e.type == "circle.brain.asked"]


def test_the_box_live_event_log_is_never_written(tmp_path):
    w = World(tmp_path)

    code, report = cli(w, *ask_args(tmp_path / "data" / "events", "--opt-in"))

    assert (code, report["kind"]) == (2, "usage")
    assert w.fake.requests == [] and not (tmp_path / "data" / "events").exists()


def test_an_account_outside_the_teachers_role_is_refused(tmp_path):
    w = World(tmp_path)

    code, report = cli(w, "status", "--account", "D")

    assert (code, report["kind"]) == (2, "pod_role") and "pod-b" in report["error"]
    assert w.fake.requests == []


def test_missing_keys_are_exit_2_and_name_the_keys_not_their_values(tmp_path):
    w = World(tmp_path)

    code, report = cli(w, "status", environ={"LAB_POD_ROLES": "C:pod-a,D:pod-b"})

    assert (code, report["kind"], report["missing"]) == (2, "missing_keys", ["AUP_HUB_USER_C", "AUP_HUB_TOKEN_C"])


def test_a_stopped_pod_is_exit_1_with_its_kind(tmp_path):
    w = World(tmp_path)
    w.fake.stop_server()

    status = cli(w, "status")
    asked = cli(w, *ask_args(tmp_path / "log", "--opt-in"))

    assert (status[0], status[1]["kind"]) == (1, "pod_stopped")
    assert (asked[0], asked[1]["kind"], asked[1]["ask_id"]) == (1, "pod_stopped", None)
    assert [e.type for e in logged(tmp_path / "log")] == ["circle.brain.consent.changed"]
