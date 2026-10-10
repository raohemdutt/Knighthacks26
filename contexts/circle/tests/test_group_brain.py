"""The group brain (G4): opt-in, text only, every send in the log before it leaves, one question at a time.

Every test runs the real GroupBrain, the real TeacherLink over lab/proxy.py's PodLink and the stock runtime client, and
the real EventBus writing its log to a temporary folder. The only double is the network: lab/teacher_fake.py's
FakeTeacher, which answers as vLLM does behind the hub proxy (the token, the browser agent, streaming, a stopped pod's
424), on a fake clock."""

import asyncio
import threading

import pytest

from bus import EventBus
from contexts.circle.application.group_brain import (BADGE, SYSTEM, BrainAnswer, BrainRefused, BrainUnavailable,
                                                     GroupBrain)
from contexts.circle.infrastructure.teacher import TeacherLink
from contexts.circle.tests.brain_helpers import (ENV, FACTS, PLAN, QUESTION, TOKEN, USER, USER_TEXT, World, opted_in,
                                                 run)
from contracts.circle import BrainConsentChanged, BrainText
from lab.hub import MissingCredentials, PodRoleError
from lab.teacher_fake import Turn


# --- opt-in -------------------------------------------------------------------------------------------------------

def test_nothing_leaves_and_nothing_is_logged_before_the_member_opts_in(tmp_path):
    w = World(tmp_path)

    with pytest.raises(BrainRefused) as refused:
        run(w.brain.ask(QUESTION, FACTS))

    assert refused.value.reason == "not_opted_in"
    assert w.fake.requests == [] and w.logged() == []
    assert w.brain.status()["opted_in"] is False


def test_the_opt_in_is_an_event_in_the_log_and_every_ask_names_it(tmp_path):
    w, consent = opted_in(tmp_path)

    answer = run(w.brain.ask(QUESTION, FACTS))

    [logged_consent, asked, _] = w.logged()
    assert logged_consent == consent
    assert (consent.circle_id, consent.member_id, consent.on, consent.source) == ("team", "ilan", True, "phone")
    assert asked.opt_in_ref == answer.opt_in_ref == consent.consent_id
    assert w.brain.status() == {"circle_id": "team", "member_id": "ilan", "opted_in": True,
                                "opt_in_ref": consent.consent_id, "changed_at": consent.ts.isoformat(),
                                "node": "pod-a", "host": "teacher.test", "badge": BADGE}


def test_opting_out_stops_the_next_question_and_both_changes_outlive_a_restart(tmp_path):
    w, on = opted_in(tmp_path)

    restarted = GroupBrain("team", "ilan", w.link, EventBus(w.log_dir).publish)
    replay = EventBus(w.log_dir)
    replay.subscribe("circle.brain.consent", restarted.on_consent, replay=True)
    run(replay.replay_into())
    assert restarted.opt_in_ref == on.consent_id

    off = run(restarted.set_consent(False, "phone"))
    with pytest.raises(BrainRefused) as refused:
        run(restarted.ask(QUESTION, FACTS))

    assert refused.value.reason == "not_opted_in"
    assert w.chats() == []
    again = GroupBrain("team", "ilan", w.link, EventBus(w.log_dir).publish)
    replay = EventBus(w.log_dir)
    replay.subscribe("circle.brain.consent", again.on_consent, replay=True)
    run(replay.replay_into())
    assert again.opt_in_ref is None and again.status()["changed_at"] == off.ts.isoformat()


def test_another_members_or_another_circles_opt_in_does_not_open_this_box(tmp_path):
    w = World(tmp_path)
    for circle_id, member_id in (("team", "maya"), ("roommates", "ilan")):
        run(w.bus.publish(BrainConsentChanged(consent_id=f"{circle_id}-{member_id}", circle_id=circle_id,
                                              member_id=member_id, on=True, source="phone")))

    with pytest.raises(BrainRefused) as refused:
        run(w.brain.ask(QUESTION, FACTS))

    assert refused.value.reason == "not_opted_in" and w.fake.requests == []


# --- every send in the log, first -----------------------------------------------------------------------------------

def test_an_ask_is_in_the_log_before_it_leaves_and_its_answer_is_logged_after(tmp_path):
    w, consent = opted_in(tmp_path)

    answer = run(w.brain.ask(QUESTION, FACTS, purpose="where_to_go"))

    [_, asked, answered] = w.logged()
    assert w.asked_on_disk == [[asked]]          # on disk by the time the request reached the teacher
    [chat] = w.chats()
    assert [BrainText(**message) for message in chat.body["messages"]] == asked.messages   # exactly what left
    assert asked.messages == [BrainText(role="system", content=SYSTEM), BrainText(role="user", content=USER_TEXT)]
    assert asked.chars == len(SYSTEM) + len(USER_TEXT)
    assert (asked.circle_id, asked.member_id, asked.purpose, asked.node, asked.host, asked.model, asked.priority,
            asked.max_tokens) == ("team", "ilan", "where_to_go", "pod-a", "teacher.test", "group-brain", -10, 256)
    assert (answered.ask_id, answered.circle_id, answered.member_id, answered.ok, answered.answer, answered.error,
            answered.model) == (asked.ask_id, "team", "ilan", True, PLAN, None, "group-brain")
    assert answer == BrainAnswer(text=PLAN, ask_id=asked.ask_id, opt_in_ref=consent.consent_id, model="group-brain",
                                 latency_ms=answered.latency_ms)
    assert answered.latency_ms == round(w.clock.elapsed * 1000) and answered.latency_ms > 0
    assert answer.badge == "Asked the group brain"


def test_when_the_log_cannot_be_written_the_question_does_not_leave(tmp_path):
    w, _ = opted_in(tmp_path)

    async def failing_log(event):
        if event.type == "circle.brain.asked":
            raise OSError("No space left on device")
        await w.bus.publish(event)

    w.brain.publish = failing_log
    with pytest.raises(OSError):
        run(w.brain.ask(QUESTION, FACTS))

    assert w.chats() == []
    assert [e.type for e in w.logged()] == ["circle.brain.consent.changed"]


# --- what the request carries -------------------------------------------------------------------------------------

def test_the_request_is_plain_text_at_planning_priority_with_thinking_off_and_a_token_cap(tmp_path):
    w, _ = opted_in(tmp_path)

    answer = run(w.brain.ask(QUESTION, FACTS))

    [chat] = w.chats()
    assert chat.path == f"/user/{USER}/proxy/8000/v1/chat/completions"
    assert (chat.headers["authorization"], chat.headers["accept"]) == (f"token {TOKEN}", "text/event-stream")
    body = chat.body
    assert (body["model"], body["priority"], body["max_tokens"], body["chat_template_kwargs"], body["stream"]) == (
        "group-brain", -10, 256, {"enable_thinking": False}, True)
    assert body["messages"] == [{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER_TEXT}]
    assert "tools" not in body
    assert answer.text == PLAN and "<think>" not in answer.text   # the fake writes a thinking block unless it is off


def test_on_a_server_that_answers_only_to_teacher_it_asks_teacher_and_lists_the_models_once(tmp_path):
    w, _ = opted_in(tmp_path, aliases=())

    first = run(w.brain.ask(QUESTION, FACTS))
    second = run(w.brain.ask("And for lunch tomorrow?", ["Maya is free from noon"]))

    assert (first.model, second.model) == ("teacher", "teacher")
    assert [chat.body["model"] for chat in w.chats()] == ["teacher", "teacher"]
    assert [e.model for e in w.logged() if e.type == "circle.brain.asked"] == ["teacher", "teacher"]
    assert len(w.model_lists()) == 1


@pytest.mark.parametrize("question, facts", [
    (QUESTION, ["Maya sent a photo: data:image/jpeg;base64,/9j/4AAQSkZJRgABAQ=="]),
    ("What is in data:image/png;base64,iVBORw0KGgo= ?", []),
    (QUESTION, [b"\xff\xd8\xff\xe0 a jpeg"]),
    (QUESTION, [{"frame": "frames:cam-desk:1728627120000"}]),
    (b"Where should we meet?", []),
])
def test_anything_but_plain_text_is_refused_before_anything_is_sent_or_logged(tmp_path, question, facts):
    w, _ = opted_in(tmp_path)

    with pytest.raises(BrainRefused) as refused:
        run(w.brain.ask(question, facts))

    assert refused.value.reason == "not_text"
    assert w.fake.requests == []
    assert [e.type for e in w.logged()] == ["circle.brain.consent.changed"]


@pytest.mark.parametrize("question, facts, reason", [
    ("   ", FACTS, "empty"),
    (QUESTION, ["x" * 6000], "too_long"),
    (QUESTION, [f"fact {n}" for n in range(41)], "too_long"),
])
def test_an_empty_or_oversized_question_is_refused_before_anything_leaves(tmp_path, question, facts, reason):
    w, _ = opted_in(tmp_path)

    with pytest.raises(BrainRefused) as refused:
        run(w.brain.ask(question, facts))

    assert refused.value.reason == reason
    assert w.fake.requests == []


def test_a_single_fact_given_as_a_string_is_one_fact(tmp_path):
    w, _ = opted_in(tmp_path)

    run(w.brain.ask(QUESTION, "Maya is at the Library until 9:00 PM"))

    [chat] = w.chats()
    assert chat.body["messages"][1]["content"] == (
        "Facts:\n- Maya is at the Library until 9:00 PM\n\nQuestion: Where should we meet tonight?")


# --- one at a time ----------------------------------------------------------------------------------------------------

def test_one_question_at_a_time_and_a_second_is_refused_not_queued(tmp_path):
    arrived, release = threading.Event(), threading.Event()

    def slow(body):
        arrived.set()
        release.wait(10)
        return Turn(content=PLAN)

    w, _ = opted_in(tmp_path, policy=slow)

    async def scenario():
        first = asyncio.create_task(w.brain.ask(QUESTION, FACTS))
        assert await asyncio.to_thread(arrived.wait, 10)
        try:
            await w.brain.ask("And for lunch?", [])
        except BrainRefused as exc:
            refused = exc
        else:
            refused = None
        release.set()
        return refused, await first

    refused, answer = run(scenario())

    assert refused is not None and refused.reason == "busy"
    assert answer.text == PLAN and len(w.chats()) == 1
    assert run(w.brain.ask("And for lunch?", [])).text == PLAN       # free again once the first is answered
    assert [e.type for e in w.logged()].count("circle.brain.asked") == 2


# --- failures -------------------------------------------------------------------------------------------------------

def test_a_stopped_pod_is_found_before_anything_of_the_members_leaves(tmp_path):
    w, _ = opted_in(tmp_path)
    w.fake.stop_server()

    with pytest.raises(BrainUnavailable) as failed:
        run(w.brain.ask(QUESTION, FACTS))

    assert (failed.value.kind, failed.value.ask_id) == ("pod_stopped", None)
    assert w.chats() == [] and [e.type for e in w.logged()] == ["circle.brain.consent.changed"]


@pytest.mark.parametrize("fault, kind", [
    (lambda fake: fake.fail(500, message="EngineCore encountered an issue"), "model_server_error"),
    (lambda fake: fake.drop(), "unreachable"),
    (lambda fake: fake.cut(), "stream_interrupted"),
    (lambda fake: fake.break_stream(), "model_server_error"),
    (lambda fake: fake.stop_server(), "pod_stopped"),
])
def test_a_failure_after_the_question_left_is_logged_and_raised_for_the_caller_to_fall_back(tmp_path, fault, kind):
    w, _ = opted_in(tmp_path)
    run(w.brain.ask(QUESTION, FACTS))           # the served name is known from here on
    fault(w.fake)

    with pytest.raises(BrainUnavailable) as failed:
        run(w.brain.ask("And for lunch tomorrow?", ["Maya is free from noon"]))

    asked, answered = w.logged()[-2:]
    assert (asked.type, answered.type) == ("circle.brain.asked", "circle.brain.answered")
    assert (failed.value.kind, failed.value.ask_id) == (kind, asked.ask_id)
    assert (answered.ask_id, answered.ok, answered.answer, answered.error) == (asked.ask_id, False, None, kind)
    assert TOKEN not in str(failed.value) and USER not in str(failed.value)


def test_an_empty_answer_is_a_failure_not_an_answer(tmp_path):
    w, _ = opted_in(tmp_path, answer="")

    with pytest.raises(BrainUnavailable) as failed:
        run(w.brain.ask(QUESTION, FACTS))

    answered = w.logged()[-1]
    assert failed.value.kind == "empty" and (answered.ok, answered.error) == (False, "empty")


# --- which pod -------------------------------------------------------------------------------------------------------

def test_only_the_account_in_the_teachers_role_is_ever_asked():
    with pytest.raises(PodRoleError, match="pod-b"):
        TeacherLink.from_env(letter="D", environ=ENV, env_files=[])
    with pytest.raises(PodRoleError, match="exactly one account in the pod-a role"):
        TeacherLink.from_env(environ={**ENV, "LAB_POD_ROLES": "D:pod-b"}, env_files=[])
    with pytest.raises(MissingCredentials) as missing:
        TeacherLink.from_env(environ={"LAB_POD_ROLES": "C:pod-a,D:pod-b"}, env_files=[])
    assert missing.value.missing == ["AUP_HUB_USER_C", "AUP_HUB_TOKEN_C"]

    link = TeacherLink.from_env(environ=ENV, env_files=[])
    assert (link.node, link.host, link.link.letter) == ("pod-a", "teacher.test", "C")
    link.close()


def test_purpose_is_a_short_word():
    brain = GroupBrain("team", "ilan", TeacherLink.from_env(environ=ENV, env_files=[]), EventBus(persist=False).publish)

    with pytest.raises(ValueError, match="purpose"):
        run(brain.ask(QUESTION, FACTS, purpose="Where To Go?"))
