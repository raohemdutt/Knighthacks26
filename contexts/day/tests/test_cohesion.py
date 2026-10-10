"""The seams between the Day loop and its neighbours: GET /api/day/coach carries the "Was that right?" view, and
the evening review (its text and its narration) mentions the day's sorting when Waste Watch has a clause for it.
With neither wired in, every answer is what it was before."""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from contexts.day.api import day_router
from contexts.day.domain.narration import check, render
from contexts.day.tests.test_coach_feedback import NOW, Box
from contexts.day.tests.test_day import SUNDAY, FakeHabits, at, items, make, places  # noqa: F401

CLAUSE = "3 of 4 things you threw away went in the right bin; the food container belonged in trash"


def coach_api(box: Box, **kw) -> TestClient:
    app = FastAPI()
    app.include_router(day_router(box.loop, memory=box.loop.memory, clock=lambda: NOW, **kw))
    return TestClient(app)


def test_the_coach_endpoint_carries_the_feedback_view_when_it_is_wired(places, tmp_path):  # noqa: F811
    box = Box(places, tmp_path).live()
    box.answer("b0", right=True)
    plain = coach_api(box).get("/api/day/coach").json()
    wired = coach_api(box, feedback=box.feedback).get("/api/day/coach").json()
    assert plain["feedback"] is None
    assert wired["feedback"] == box.http.get("/api/day/coach/feedback?days=3").json()  # the same view, in one request
    assert wired["feedback"]["sessions"] and wired["feedback"]["options"]
    # additive: nothing else about the answer moved
    assert {k: v for k, v in wired.items() if k != "feedback"} == {k: v for k, v in plain.items() if k != "feedback"}


def test_the_evening_review_mentions_the_days_sorting_only_when_there_is_a_clause(items, places, tmp_path):  # noqa: F811
    habits = FakeHabits(sleep=6.35, source="camera", meals=2, focus=2.5)
    loop = make(items, places, tmp_path, habits=habits)
    p = loop.planner
    p.on_plan_proposed(p.plan(at(8, 10))[0])
    before = p.review(SUNDAY, at(20, 0))
    asked: list = []

    def nothing(day, now):
        asked.append((day, now))
        return None

    p.waste_clause = nothing  # wired, but nothing was thrown away: byte for byte the same review
    assert p.review(SUNDAY, at(20, 0)) == before and asked == [(SUNDAY, at(20, 0))]
    assert loop.narrator.review_facts(before) == {k: v for k, v in loop.narrator.review_facts(before).items() if k != "waste"}

    p.waste_clause = lambda day, now: CLAUSE + "."
    after = p.review(SUNDAY, at(20, 0))
    assert after.text == f"{before.text} Waste Watch: {CLAUSE}."
    assert after.model_copy(update={"text": before.text}) == before  # only the text changed

    facts = loop.narrator.review_facts(after)
    assert facts["waste"] == CLAUSE
    said, plain = render(facts), render({k: v for k, v in facts.items() if k != "waste"})
    assert "On sorting: three of four things you threw away went in the right bin; the food container belonged in trash." in said.spoken
    assert f"Waste Watch: {CLAUSE}." in said.display
    assert said.spoken.replace(" On sorting: three of four things you threw away went in the right bin; the food container belonged in trash.", "") == plain.spoken
    assert said.display.replace(f" Waste Watch: {CLAUSE}.", "") == plain.display
    # the clause is a fact, so a model's rewrite that says it is grounded, and one that changes it is not
    rewrite = said.spoken.replace("That's the day.", "So, that's the day.")
    assert check(rewrite, facts, said.spoken) == []
    assert check(rewrite.replace("three of four", "three of nine"), facts, said.spoken)

    def broken(day, now):
        raise RuntimeError("the impact story is not ready")

    p.waste_clause = broken  # a failing neighbour never stops the review
    assert p.review(SUNDAY, at(20, 0)) == before
