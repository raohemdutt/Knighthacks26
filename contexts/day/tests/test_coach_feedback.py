"""The coach's feedback loop: an answer about a break is an event, the review and the precision counts follow it,
enough answers propose a teach task as an approval card (never approved here), and a deployed model is noted once."""

import asyncio
from datetime import timedelta, timezone

from fastapi import FastAPI
from fastapi.testclient import TestClient

from bus import EventBus
from contexts.day.application.coach_feedback import TASK_PREFIX, TEACHABLE, CoachFeedback, coach_feedback_router
from contexts.day.tests.test_coach import block, desk, plan_with, scene
from contexts.day.tests.test_day import SUNDAY, FakeHabits, at, make, places  # noqa: F401
from contexts.learning.application.board import LearningBoard
from contexts.workspace.domain.approvals import Approvals
from contracts.day import DayCoachCorrected
from contracts.learning import ModelDeployed, RunEvaluated, RunProgress, TaskCreated, TaskSpec
from contracts.memory import Episode

HOURS = (9, 10, 11, 12, 13, 14)  # six half-hour blocks; in each the person leaves at ten past for five minutes
PHONE_AT = (9, 10, 11, 12, 13)  # ... right after picking up the phone, except in the last block
NOW = at(15)


def kf(hour: int, minute: int) -> str:
    return f"frames:cam-desk:{int(at(hour, minute).timestamp() * 1000)}"


EPISODES = [Episode(id=f"ep-desk-{h}", camera_id="cam-desk", start=at(h), end=at(h, 30), labels=["person"],
                    captions=["A person types.", "The person picks up their phone.", "A person types."],
                    keyframe_refs=[kf(h, 0), kf(h, 9), kf(h, 20)]) for h in HOURS]
DAY = sorted([e for i, h in enumerate(HOURS) for e in desk([(at(h), at(h, 10)), (at(h, 15), at(h, 30))], track=10 * (i + 1))]
             + [scene(at(h, 9), "The person picks up their phone.") for h in PHONE_AT], key=lambda e: e.ts)
PLAN = plan_with(*[block(at(h), 30, id=f"b{i}") for i, h in enumerate(HOURS)])


class Box:
    """The Day loop, the feedback loop, the learning board and the approvals on one bus."""

    def __init__(self, places, tmp_path, name="a", episodes=EPISODES, **kw):  # noqa: F811
        self.bus = EventBus(tmp_path / "events")
        self.loop = make([], places, tmp_path / name, habits=FakeHabits(focus=1.2), episodes=episodes)
        self.loop.subscribe(self.bus)
        self.board = LearningBoard()
        self.board.subscribe(self.bus)
        self.approvals = Approvals({})
        self.bus.subscribe("agent.action.proposed", self.approvals.on_proposed, replay=True)
        self.feedback = CoachFeedback(self.loop.coach, memory=self.loop.memory, learning=self.board, **kw)
        self.feedback.subscribe(self.bus)
        app = FastAPI()
        app.include_router(coach_feedback_router(self.loop, self.feedback, publish=self.bus.publish, clock=lambda: NOW))
        self.http = TestClient(app)

    def live(self) -> "Box":
        async def run():
            await self.bus.publish(PLAN)
            for e in DAY:
                await self.bus.publish(e)

        asyncio.run(run())
        return self

    def answer(self, block_id: str, right: bool, answer: str | None = None, note: str | None = None, index: int = 0) -> dict:
        r = self.http.post("/api/day/coach/corrections", json={"day": SUNDAY.isoformat(), "block_id": block_id, "break_index": index,
                                                               "right": right, "answer": answer, "note": note})
        assert r.status_code == 200, r.text
        return r.json()

    def log(self, prefix: str) -> list:
        return [e for e in self.bus.read_log() if e.type.startswith(prefix)]


def test_the_scripted_day_has_five_phone_breaks_and_one_with_no_visible_reason(places, tmp_path):  # noqa: F811
    box = Box(places, tmp_path).live()
    view = box.http.get("/api/day/coach/feedback").json()
    sessions = {s["block_id"]: s for s in view["sessions"]}
    assert [sessions[f"b{i}"]["breaks"][0]["guess"] for i in range(6)] == ["phone"] * 5 + ["none"]
    b0 = sessions["b0"]["breaks"][0]
    assert (b0["index"], b0["minutes"], b0["correction"], b0["frame_ref"]) == (0, 5, None, kf(9, 9))  # the picture of that moment
    assert b0["guess_text"] == "right after the phone came into view"
    assert [o["id"] for o in view["options"]] == ["phone", "chat", "door", "food", "bathroom", "done", "other"]
    assert view["precision"] == [] and view["learning"] == [] and view["notes"] == [] and view["answers"] == 0
    assert view["sessions"][0]["block_id"] == "b5"  # newest first


def test_an_answer_is_an_event_and_the_review_and_the_suggestions_reflect_it(places, tmp_path):  # noqa: F811
    box = Box(places, tmp_path).live()
    before = box.http.get("/api/day/coach/feedback").json()
    assert "Keep the phone face down" in before["suggestions"][0]  # five phone breaks: the coach blames the phone

    got = box.answer("b0", right=False, answer="bathroom", note="  quick break  ")
    (e,) = box.log("day.coach.corrected")
    assert isinstance(e, DayCoachCorrected)
    assert (e.day, e.block_id, e.break_index, e.break_start, e.guess, e.right, e.answer, e.note, e.plan_id) == (
        SUNDAY, "b0", 0, at(9, 10), "phone", False, "bathroom", "quick break", PLAN.plan_id)
    assert set(e.model_dump()) == set(DayCoachCorrected.model_fields) and "frame" not in e.model_dump_json()  # ids and times only
    b = got["session"]["breaks"][0]
    assert (b["guess"], b["trigger"]) == ("phone", "none")  # the coach's guess is kept; the reason is now the person's
    assert b["correction"]["answer_text"] == "a bathroom break" and b["correction"]["note"] == "quick break"
    assert got["session"]["text"].endswith("You said the 9:10 AM break was a bathroom break, not the phone.")
    assert got["proposed"] is None

    for i in (1, 2, 3, 4):  # the person says none of them was the phone: the phone advice goes away
        box.answer(f"b{i}", right=False, answer="done")
    after = box.http.get("/api/day/coach/feedback").json()
    assert not any("phone" in s for s in after["suggestions"]) and after["answers"] == 5
    # a second answer about the same break replaces the first
    box.answer("b0", right=True)
    assert box.feedback.corrections[SUNDAY, "b0", at(9, 10).astimezone(timezone.utc).isoformat()].right is True
    assert len(box.feedback.corrections) == 5
    # "yes" needs no answer; "no" does; an unknown break is a 404
    assert box.http.post("/api/day/coach/corrections", json={"day": "2026-10-11", "block_id": "b1", "break_index": 0, "right": False}).status_code == 422
    assert box.http.post("/api/day/coach/corrections", json={"day": "2026-10-11", "block_id": "b1", "break_index": 3, "right": True}).status_code == 404
    assert box.http.post("/api/day/coach/corrections", json={"day": "2026-10-11", "block_id": "nope", "break_index": 0, "right": True}).status_code == 404


def test_precision_is_counted_per_trigger_from_the_answers(places, tmp_path):  # noqa: F811
    box = Box(places, tmp_path, min_corrections=99).live()
    for i in (0, 1, 2, 3):
        box.answer(f"b{i}", right=True)
    box.answer("b4", right=False, answer="chat")
    box.answer("b5", right=True)  # "no visible reason" was right
    rows = {r["trigger"]: r for r in box.http.get("/api/day/coach/feedback").json()["precision"]}
    assert rows["phone"] == {"trigger": "phone", "right": 4, "total": 5, "precision": 0.8,
                             "text": "The coach is right about phone breaks 4 of 5 times."}
    assert rows["none"]["text"] == "The coach is right about breaks with no visible reason 1 of 1 time."
    assert "chat" not in rows  # the coach never guessed chat: an answer of "chat" says nothing about its chat guesses


def test_a_restart_rebuilds_the_answers_and_the_corrected_review_from_the_log(places, tmp_path):  # noqa: F811
    box = Box(places, tmp_path, min_corrections=99).live()
    box.answer("b0", right=True)
    box.answer("b1", right=False, answer="food", note="grabbed a coffee")
    box.answer("b5", right=False, answer="phone")
    again = Box(places, tmp_path, name="b", min_corrections=99)
    asyncio.run(again.bus.replay_into())
    assert again.feedback.corrections == box.feedback.corrections and len(again.feedback.corrections) == 3
    assert again.feedback.view(NOW) == box.feedback.view(NOW)
    assert again.feedback.precision() == box.feedback.precision()
    s = next(s for s in again.feedback.view(NOW)["sessions"] if s["block_id"] == "b1")
    assert s["breaks"][0]["trigger"] == "food" and "was food or a drink, not the phone" in s["text"]


def test_enough_answers_propose_one_teach_task_as_an_approval_card_and_never_approve_it(places, tmp_path):  # noqa: F811
    box = Box(places, tmp_path).live()
    assert box.feedback.min_corrections == 5 and box.feedback.min_negatives == 2  # the defaults
    box.answer("b0", right=True)
    box.answer("b1", right=False, answer="bathroom")
    box.answer("b5", right=False, answer="phone")  # the coach saw no reason; the person says it was the phone
    got = box.answer("b2", right=True)
    assert got["proposed"] is None and box.log("learning.") == [] and box.log("agent.action") == []  # four answers, one "no"
    assert got["progress"] == [{"trigger": "phone", "name": "phone in hand", "answers": 4, "negatives": 1, "needed": 5,
                                "negatives_needed": 2, "ready": False, "proposed": False}]

    got = box.answer("b3", right=False, answer="done")  # the fifth answer and the second "no"
    (created,) = box.log("learning.task.created")
    (card,) = box.log("agent.action.proposed")
    assert isinstance(created, TaskCreated) and created.task == TASK_PREFIX + "phone" == "desk-distraction-phone"
    assert created.instruction == "Is a phone in the person's hand or being looked at? Answer yes or no and say where it is."
    assert (card.kind, card.target) == ("export_training_data", "desk-distraction-phone") and got["proposed"] == {"phone": card.action_id}
    assert card.reason.startswith("Teach Atlas to spot 'phone in hand' from 5 of your own blurred examples? "
                                  "(3 where you said that was it, 2 where you said it was not.)")
    assert "to the lab" in card.reason
    spec = TaskSpec.model_validate(card.params["spec"])
    assert spec.camera == "cam-desk" and spec.output_schema["properties"]["present"] == {"type": "boolean"}
    verdicts = {x.frame_ref: x.verdict for x in spec.examples}
    assert verdicts == {kf(9, 9): "up", kf(11, 9): "up",  # confirmed
                        kf(14, 9): "up",  # the coach missed it and the person said phone: the keyframe nearest the break
                        kf(10, 9): "down", kf(12, 9): "down"}  # the coach said phone and the person said no
    assert card.params["end"] == NOW.isoformat() and card.params["start"] == (NOW - timedelta(hours=24)).isoformat()

    # it is the normal approval card, still waiting; nothing was approved, exported or decided by anyone
    (pending,) = box.approvals.pending()
    assert pending.id == card.action_id and pending.status == "proposed"
    assert box.log("experience.approval") == [] and box.log("learning.dataset") == [] and box.log("workspace.action.completed") == []
    view = box.http.get("/api/day/coach/feedback").json()
    (row,) = view["learning"]
    assert (row["trigger"], row["task"], row["stage"], row["learned"]) == ("phone", "desk-distraction-phone", "waiting for approval", False)
    assert "waiting for your OK" in row["text"] and "Nothing has left the box" in row["text"] and view["notes"] == []

    # more answers do not propose it again, live or after a restart
    assert box.answer("b4", right=True)["proposed"] is None
    again = Box(places, tmp_path, name="b")
    asyncio.run(again.bus.replay_into())
    assert again.feedback.tasks.keys() == {"phone"} and again.feedback.view(NOW)["learning"] == box.feedback.view(NOW)["learning"]
    again.answer("b4", right=False, answer="chat")
    assert len(again.log("learning.task.created")) == 1 and len(again.log("agent.action.proposed")) == 1


def test_the_threshold_is_configurable_and_needs_its_negatives_and_a_picture(places, tmp_path):  # noqa: F811
    box = Box(places, tmp_path, min_corrections=3, min_negatives=1).live()
    box.answer("b0", right=True)
    box.answer("b1", right=True)
    assert box.answer("b2", right=True)["proposed"] is None  # three answers, but nobody ever said no
    assert box.answer("b3", right=False, answer="food")["proposed"]  # the first "no": now there is something to learn from
    (card,) = box.log("agent.action.proposed")
    assert card.reason.startswith("Teach Atlas to spot 'phone in hand' from 4 of your own blurred examples?")
    assert box.log("learning.task.created")[0].task == "desk-distraction-phone"  # "food" has one answer: no task for it

    # a box whose Memory holds no picture of those breaks has nothing to teach from, and proposes nothing
    blind = Box(places, tmp_path / "blind", episodes=[], min_corrections=2, min_negatives=1).live()
    blind.answer("b0", right=True)
    assert blind.answer("b1", right=False, answer="done")["proposed"] is None
    assert blind.log("learning.") == [] and blind.log("agent.action") == []
    assert blind.feedback.progress()[0]["ready"] is True and blind.feedback.progress()[0]["proposed"] is False
    # chat is not something a camera can be taught to see: no task however many answers
    assert "chat" not in TEACHABLE


def test_when_the_model_for_that_task_is_deployed_the_coach_notes_it_once(places, tmp_path):  # noqa: F811
    box = Box(places, tmp_path, min_corrections=2, min_negatives=1).live()
    box.answer("b0", right=True)
    box.answer("b1", right=False, answer="done")
    slug = "desk-distraction-phone"

    async def lab():
        await box.bus.publish(RunProgress(ts=NOW + timedelta(hours=1), task=slug, run_id="r1", stage="training", progress=0.4))

    asyncio.run(lab())
    (row,) = box.feedback.view(NOW)["learning"]
    assert row["stage"] == "training" and row["text"] == "Atlas is learning this: phone in hand (training)." and not row["learned"]
    assert box.feedback.view(NOW)["notes"] == []

    async def deployed():
        await box.bus.publish(RunEvaluated(ts=NOW + timedelta(hours=2), task=slug, run_id="r1", decision="deploy",
                                           scores={"current": 0.41, "candidate": 0.78}, model_version="atlas-v4-phone"))
        await box.bus.publish(ModelDeployed(ts=NOW + timedelta(hours=3), model_version="atlas-v4-phone", previous_version="atlas-base"))
        await box.bus.publish(ModelDeployed(ts=NOW + timedelta(hours=4), model_version="atlas-v5-other", previous_version="atlas-v4-phone"))

    asyncio.run(deployed())
    view = box.http.get("/api/day/coach/feedback").json()
    (row,) = view["learning"]
    assert row["learned"] and row["stage"] == "done" and row["model_version"] == "atlas-v4-phone"
    assert view["notes"] == ["Atlas now recognises phone in hand; break reasons should be more accurate."]  # once, not per break
    # a restarted box says the same, from the log alone
    again = Box(places, tmp_path, name="b", min_corrections=2, min_negatives=1)
    asyncio.run(again.bus.replay_into())
    assert again.feedback.view(NOW)["notes"] == view["notes"]
    # a model deployed for some other task is not the coach's to note
    other = Box(places, tmp_path / "other").live()
    asyncio.run(other.bus.publish(TaskCreated(ts=NOW, task="desk-objects", instruction="List the objects.")))
    assert other.feedback.view(NOW)["learning"] == []
