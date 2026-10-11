"""The twelve earlier days (contexts/demo/history.py): lived through the real Day loop, read back by Today's Plan."""

import asyncio
import shutil
from datetime import datetime, timedelta

import pytest

from contexts.demo import history
from contexts.demo.__main__ import main
from contexts.demo.seed import seed
from contexts.demo.story import load
from contexts.play.dayrun.models import Move

NOW = load().now
TODAY = NOW.date()
DAY_TYPES = {"day.plan.proposed", "day.replan.proposed", "day.plan.committed", "day.reviewed", "day.status.changed", "day.narrated",
             "agent.action.proposed", "workspace.approval.requested", "workspace.action.completed", "experience.approval.decided"}


@pytest.fixture(scope="module")
def plain(tmp_path_factory):
    seeded = asyncio.run(seed(tmp_path_factory.mktemp("plain")))
    yield seeded
    seeded.app.projection.store.close()


@pytest.fixture(scope="module")
def seeded(tmp_path_factory):
    data_dir = tmp_path_factory.mktemp("history")
    out = asyncio.run(seed(data_dir, history=True))
    out.app.dayrun.clock = lambda: NOW  # "today" is the story's day, whatever the wall clock says
    yield data_dir, out
    out.app.projection.store.close()


def ago(n: int):
    return TODAY - timedelta(days=n)


def test_twelve_days_are_planned_and_reviewed_by_the_loop_and_one_is_left_empty(seeded):
    _, s = seeded
    planner = s.app.day.planner
    days = history.history_days(NOW)
    assert days == [ago(n) for n in range(15, 3, -1)] and len(days) == history.DAYS == len(history.MEASURED)
    assert all(d in planner.plans and d in planner.reviews for d in days)
    assert ago(3) not in planner.plans and ago(3) not in planner.reviews  # the gap before the story's own days
    for d in days:  # the loop's own ids, so nothing here was written by hand
        assert planner.plans[d].event_id == f"day-plan-{planner.plans[d].plan_id}" and planner.plans[d].plan_id.startswith(f"plan-{d}-08")
        assert planner.reviews[d].event_id.startswith("day-review-plan-") and planner.reviews[d].score is not None
        assert any(m.fixed for m in planner.plans[d].moves)  # each day's calendar is in its plan


def test_the_three_presets_are_days_in_the_log(seeded):
    _, s = seeded
    p = s.app.day.planner
    own, kept, changed = ago(history.OWN_PLAN), ago(history.ON_PLAN), ago(history.CHANGED)
    assert p.plans[own].by == "person" and not p.plans[own].outcome.won and p.reviews[own].stars == 0
    assert p.plans[kept].by == "atlas" and p.plans[kept].outcome.won and p.plans[kept].plan_id in p.committed and p.reviews[kept].score >= 80
    replan = p.current[changed]
    assert replan.type == "day.replan.proposed" and replan.plan_id in p.committed and replan.outcome.won
    assert replan.why.startswith("Two things changed: PHY2049 lecture moved to Engineering II and now starts at 1:30 PM")
    titles = {m.title: m for m in replan.moves if m.fixed}
    assert titles["PHY2049 lecture"].place == "ucf.eng2" and "COP4533 review session" in titles
    assert p.reviews[changed].plan_id == replan.plan_id


def test_the_story_itself_is_exactly_what_it_was(plain, seeded):
    _, s = seeded
    a, b = plain.app, s.app
    for day in (TODAY - timedelta(days=1), TODAY):
        assert a.habits.digest(day=day, now=NOW) == b.habits.digest(day=day, now=NOW)
    assert {h.id: a.habits.history_for(h.id) for h in a.habits.cfg.habits} == {h.id: b.habits.history_for(h.id) for h in b.habits.cfg.habits}
    assert a.waste.metrics() == b.waste.metrics() and a.memory.stats() == b.memory.stats()
    assert a.league.view(NOW) == b.league.view(NOW)
    for day in (ago(2), ago(1), TODAY):  # the story's own plans and reviews: same ids, same outcome
        assert a.day.planner.plans[day].model_dump(exclude={"action_id"}) == b.day.planner.plans[day].model_dump(exclude={"action_id"})
    assert a.day.planner.reviews[ago(1)] == b.day.planner.reviews[ago(1)]
    assert [(e.rule_id, e.message) for e in plain.alerts] == [(e.rule_id, e.message) for e in s.alerts]
    assert [e.message for e in plain.nudges] == [e.message for e in s.nudges] and len(plain.rounds) == len(s.rounds)
    # what it adds is the Day loop's own events and their approval cards, and nothing a camera or a connector publishes
    more = {t: n - plain.written.get(t, 0) for t, n in s.written.items() if n != plain.written.get(t, 0)}
    assert set(more) <= DAY_TYPES and more["day.plan.proposed"] == 13 and more["day.reviewed"] == 12 and more["day.plan.committed"] == 3
    assert more["day.replan.proposed"] == 1 and more["experience.approval.decided"] == 3


def test_the_date_strip_reads_them_back(seeded):
    _, s = seeded
    strip = s.app.dayrun.days()
    assert strip["today"] == TODAY and len(strip["days"]) == 20
    by = {d["day"]: d for d in strip["days"]}
    assert by[TODAY]["when"] == "today" and by[ago(3)] == {"day": ago(3), "when": "past", "has_data": False, "label": "nothing recorded", "summary": ""}
    assert all(by[d]["has_data"] and by[d]["when"] == "past" for d in history.history_days(NOW)[1:])  # the oldest is past the strip's two weeks
    own = by[ago(history.OWN_PLAN)]
    assert own["label"] == "the plan missed COP4533 lecture by 11 min" and own["won"] is False and own["stars"] == 0
    assert own["summary"] == f"Your own plan; {own['score']}% of it happened."
    assert by[ago(history.ON_PLAN)]["label"] == "on plan" and by[ago(history.ON_PLAN)]["won"]
    changed = by[ago(history.CHANGED)]
    assert changed["label"] == "re-planned, still made it" and changed["summary"].startswith("Two things changed: ")
    ahead = by[TODAY + timedelta(days=2)]  # Tuesday: the milestone is due
    assert ahead["when"] == "future" and ahead["has_data"] and [d["title"] for d in ahead["due"]] == ["Project Milestone1"]
    assert 0 < ahead["free_minutes"] <= 16 * 60 and 0 <= ahead["pressure"] <= 1
    assert not by[TODAY + timedelta(days=5)]["has_data"]  # nothing on it yet: not selectable


def test_a_past_day_is_its_plan_with_what_the_review_measured_and_can_be_run_again(seeded):
    _, s = seeded
    dayrun = s.app.dayrun
    own = dayrun.day(ago(history.OWN_PLAN))
    assert own["when"] == "past" and own["review"]["score"] == own["score"] and own["review"]["change_tomorrow"]
    assert [b["title"] for b in own["blocks"] if b["fixed"]] == ["COP4533 lecture", "PHY2049 lecture"]
    assert any(b["mark"] == "missed" for b in own["blocks"]) and not any(b["moved"] for b in own["blocks"])
    # the same plan, run again on that day's board: the lecture is there, and the lunch still misses it by 11 minutes
    start, moves = datetime.fromisoformat(own["start"]), [Move(**m) for m in own["moves"]]
    again = dayrun.simulate(moves, start)
    assert [b.title for b in again.blocks if b.fixed] == ["COP4533 lecture", "PHY2049 lecture"] and again.rejected[0].minutes == 11
    fix = dayrun.explain(moves, start)["fixes"][0]
    assert fix.problem == "You'd miss COP4533 lecture by 11 minutes" and fix.mends

    changed = dayrun.day(ago(history.CHANGED))
    assert changed["first"] and changed["changes"][0]["why"].startswith("Two things changed") and changed["won"]
    titles = [b["title"] for b in changed["blocks"]]
    assert titles.index("COP4533 lecture") < titles.index("COP4533 review session") < titles.index("PHY2049 lecture")  # the morning, then the new plan
    moved = {b["title"] for b in changed["blocks"] if b["moved"]}
    assert {"COP4533 review session", "PHY2049 lecture"} <= moved and "COP4533 lecture" not in moved
    rerun = dayrun.simulate([Move(**m) for m in changed["moves"]], datetime.fromisoformat(changed["start"]))
    assert rerun.outcome.won and not rerun.rejected

    today = dayrun.day(TODAY)
    assert today["when"] == "today" and today["moves"] and today["start"] == s.app.day.planner.current[TODAY].start.isoformat()
    with pytest.raises(LookupError):
        dayrun.day(ago(3))


def test_seeding_the_history_twice_adds_nothing(seeded, tmp_path):
    data_dir, _ = seeded
    copy = tmp_path / "data"
    shutil.copytree(data_dir, copy)
    before = sorted((p.name, p.read_bytes()) for p in (copy / "events").iterdir())
    again = asyncio.run(seed(copy, force=True, history=True))
    again.app.projection.store.close()
    assert not again.written
    assert sorted((p.name, p.read_bytes()) for p in (copy / "events").iterdir()) == before


def test_the_plain_seed_has_no_history_and_the_flag_asks_for_it(plain, tmp_path, capsys):
    assert not any(d in plain.app.day.planner.plans for d in history.history_days(NOW))
    with pytest.raises(SystemExit):
        main(["seed", "--data-dir", str(tmp_path / "x"), "--histry"])
    capsys.readouterr()
