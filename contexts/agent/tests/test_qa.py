from datetime import UTC, datetime, timedelta

from contexts.agent.application.memory_port import FakeMemory
from contexts.agent.application.qa import NO_EVIDENCE, QA
from contracts.experience import QuestionAsked
from contracts.memory import Episode, Sighting
from contracts.types import ChatResult, ToolCall
from runtime import FakeRuntime

NOW = datetime(2026, 10, 11, 9, 0, tzinfo=UTC)


def ep(id, hours_ago, captions, camera="cam-desk", labels=("person",)):
    start = NOW - timedelta(hours=hours_ago)
    return Episode(
        id=id,
        camera_id=camera,
        start=start,
        end=start + timedelta(minutes=5),
        labels=list(labels),
        captions=captions,
        keyframe_refs=[f"frames:{camera}:{id}"],
    )


MEMORY = FakeMemory(
    episodes=[
        ep("e1", 8, ["A person packs a laptop and leaves the table."]),
        ep("e2", 3, ["The table is empty, lights dim."]),
        ep("e3", 0.5, ["A courier drops a box at the door."], camera="cam-door", labels=("person", "package")),
        ep("e4", 30, ["Old episode from two days ago."]),
    ],
    sightings=[
        Sighting(label="keys", camera_id="cam-desk", zone_id="desk", ts=NOW - timedelta(hours=2), frame_ref="frames:cam-desk:k1", position="next to the mug"),
    ],
)


def qa(*replies):
    rt = FakeRuntime(chat_replies=[ChatResult(content=r, model_version="v1") if isinstance(r, str) else r for r in replies])
    return QA(rt, MEMORY, clock=lambda: NOW), rt


def evidence_prompt(rt):
    return rt.calls[-1][1][-1].content


def test_where_are_my_keys_prefetches_last_seen():
    q, rt = qa("Your keys are on the desk next to the mug.")
    a = q.ask("Where did I leave my keys?")
    assert a.citations[0].frame_ref == "frames:cam-desk:k1"
    assert "next to the mug" in evidence_prompt(rt)
    assert a.model_version == "v1"


def test_overnight_uses_hour_window_and_skips_old_episodes():
    q, rt = qa("Someone left, then it was quiet.")
    a = q.ask("What happened at our table overnight?")
    prompt = evidence_prompt(rt)
    assert "packs a laptop" in prompt and "two days ago" not in prompt
    assert {c.episode_id for c in a.citations} == {"e1", "e2", "e3"}


def test_last_hour_door():
    q, rt = qa("Yes, a courier dropped a box.")
    a = q.ask("Did anyone come to the door in the last hour?")
    assert [c.episode_id for c in a.citations] == ["e3"]


def test_no_evidence_does_not_call_the_model():
    q, rt = qa()
    a = q.ask("Where is my wallet?")
    assert a.text == NO_EVIDENCE
    assert not any(name == "chat" for name, _ in rt.calls)


def test_open_question_runs_the_tool_loop():
    call = ChatResult(content="", tool_calls=[ToolCall(id="c1", name="search_memory", arguments={"q": "courier box"})], model_version="v1")
    q, rt = qa(call, "A courier dropped a box at the door.")
    a = q.ask("Anything delivered?")
    assert a.text == "A courier dropped a box at the door."
    assert a.citations[0].episode_id == "e3"
    tool_msg = rt.calls[-1][1][-1]
    assert tool_msg.role == "tool" and "courier" in tool_msg.content


def test_tool_loop_without_evidence_says_so():
    q, _ = qa("Probably nothing happened.")
    assert q.ask("Anything weird?").text == NO_EVIDENCE


def test_on_question_emits_answer_event():
    q, _ = qa("On the desk.")
    out = q.on_question(QuestionAsked(question_id="q1", text="where are my keys"))
    assert (out.question_id, out.answer, out.model_version) == ("q1", "On the desk.", "v1")
