"""The chooser (contexts/day/domain/choose.py): the win rules first, then points, on real boards."""

from datetime import timedelta

import pytest

from contexts.day.domain.choose import choose
from contexts.day.tests.test_day import MILESTONE, FakeHabits, at, make
from contexts.demo.story import story
from contexts.play.dayrun.models import InitialOverride
from contexts.play.domain.places import PlaceSet
from contracts.memory import Item

JUDGING = "outlook:kh-judging"
ESSAY = Item(connector="canvas", kind="assignment", external_id="hist:essay", title="History essay", when=at(23, 30))
QUIZ = Item(connector="canvas", kind="assignment", external_id="stats:quiz", title="Stats quiz", when=at(20, 0, day=12))
STUDY_SLACK = 30  # chosen study minutes allowed beyond the focused minutes a deadline needs


def items_of(now=None) -> list[Item]:
    return [Item(**e.model_dump(include={"connector", "kind", "external_id", "title", "when", "url", "summary"}))
            for e in story(now) if e.type == "connector.item.synced"]


@pytest.fixture(scope="module")
def places() -> PlaceSet:
    return PlaceSet.load("config/places")


@pytest.fixture(scope="module")
def busy_items() -> list[Item]:
    """The demo story moved to end at 1:01 PM, as on the day this was seen: a check-in 12:51-13:11, submissions
    14:31-14:46, judging 15:31-17:31, a study session 17:31-20:31, and the milestone (300 focused minutes) due later."""
    return items_of(at(13, 1))


def busy(busy_items, places, tmp_path, meals: int = 1, when=None, **over):
    """The busy day's board at 8:16 after a four-hour night, with the Day Run service that built it."""
    dayrun = make(busy_items, places, tmp_path, habits=FakeHabits(sleep=4.0, meals=meals)).planner.dayrun
    initial = {"energy": 44, "sleep_hours": 4.0, "meals": meals, **over}
    return dayrun, dayrun.board(when or at(8, 16), InitialOverride(**initial))


def chosen(result, kind: str | None = None) -> list:
    return [b for b in result.blocks if not b.fixed and (kind is None or b.kind == kind)]


def lates(result) -> int:
    return sum(c.kind == "late" for c in result.changes)


@pytest.mark.parametrize("meals", [0, 1])  # as it was seen (the old plan: "Meals: 1 (2 needed)"), and with breakfast already eaten
def test_the_observed_busy_day_gets_a_plan_that_wins(busy_items, places, tmp_path, meals):
    dayrun, board = busy(busy_items, places, tmp_path, meals)
    cfg = dayrun.cfg
    fixed = [(b.start.strftime("%H:%M"), b.end.strftime("%H:%M")) for b in board.blocks]
    assert fixed == [("12:51", "13:11"), ("14:31", "14:46"), ("15:31", "17:31"), ("17:31", "20:31")]
    (deadline,) = board.deadlines
    assert deadline.work_left == 300 and board.initial.energy == 44 and board.initial.sleep_hours == 4.0

    moves, result = choose(cfg, dayrun.walks, board)
    assert result.outcome.won and result.outcome.reasons == [] and not result.rejected
    assert result.totals.meals >= cfg.win.meals
    assert min(t.energy for t in result.timeline) >= cfg.stars.energy_floor  # energy never crosses the floor
    assert result.totals.sleep_hours >= cfg.win.sleep_hours
    night = max(chosen(result, "sleep"), key=lambda b: b.minutes)
    assert night.minutes >= cfg.sleep.need_hours * 60 and night.start <= board.sleep_window[0]  # a short night: no later than usual

    # Study: at least the focused minutes the win needs (the deadline on track when the board ends), and no more
    # chosen study than the deadline's 300 focused minutes plus a margin. (The old plan: 687 study minutes.)
    hours_after = (deadline.due - board.end).total_seconds() / 3600
    needed = deadline.work_left - cfg.on_track_risk * hours_after * cfg.capacity_minutes_per_day / 24
    assert needed > 0 and result.totals.focused_minutes >= needed
    assert result.totals.focused_minutes <= deadline.work_left
    study = chosen(result, "study")
    assert 0 < sum(b.minutes for b in study) <= deadline.work_left + STUDY_SLACK
    assert result.totals.study_minutes < 687 and all(b.minutes <= 90 for b in study)
    assert lates(result) == lates(choose(cfg, dayrun.walks, board.model_copy(update={"deadlines": []}))[1])  # only the one no plan can avoid


def test_a_meal_displaced_by_a_fixed_block_is_moved_not_dropped(busy_items, places, tmp_path):
    dayrun, board = busy(busy_items, places, tmp_path)
    _, result = choose(dayrun.cfg, dayrun.walks, board)
    eats = chosen(result, "eat")
    # lunch at 12:30 would run into the 12:51 check-in: it is eaten before it, with the walk on to Engineering II
    lunch = next(b for b in eats if at(11, 30) <= b.start < at(12, 51))
    assert lunch.end <= at(12, 51) and lunch.start < at(12, 30) and lunch.minutes >= dayrun.cfg.meal_min_minutes
    # dinner at 18:30 falls inside judging and the study session (15:31 to 20:31): it is eaten right after
    dinner = next(b for b in eats if b.start >= at(17, 31) and b.start.date() == at(8).date())
    assert at(20, 31) <= dinner.start <= at(21, 0)
    assert result.totals.meals == board.initial.meals + len(eats) and not result.rejected  # every one of them counts as a meal


def test_an_easy_empty_day_is_not_filled_with_study(places, tmp_path):
    dayrun = make([], places, tmp_path, habits=FakeHabits(sleep=7.5, meals=1)).planner.dayrun
    moves, result = choose(dayrun.cfg, dayrun.walks, dayrun.board(at(8, 0)))
    assert result.outcome.won and result.stars == 3
    assert result.totals.study_minutes == 0 and {m.kind for m in moves} == {"eat", "sleep"}  # lunch, dinner, the night; the rest is free
    # one small deadline: the study it needs, and no more
    dayrun = make([QUIZ], places, tmp_path, habits=FakeHabits(sleep=7.5, meals=1)).planner.dayrun
    board = dayrun.board(at(8, 0))
    moves, result = choose(dayrun.cfg, dayrun.walks, board)
    (quiz,) = board.deadlines
    assert result.outcome.won and result.carry.work_left[quiz.id] == 0
    assert quiz.work_left <= result.totals.study_minutes <= quiz.work_left + STUDY_SLACK
    assert "rest" not in {m.kind for m in moves}


def test_a_hopeless_late_start_returns_a_labelled_plan_with_reasons_and_no_padding(places, tmp_path):
    loop = make([ESSAY], places, tmp_path, habits=FakeHabits(sleep=7.0, meals=1))
    dayrun = loop.planner.dayrun
    board = dayrun.board(at(21, 0))  # 300 focused minutes due at 23:30
    moves, result = choose(dayrun.cfg, dayrun.walks, board)
    assert not result.outcome.won and result.stars == 0 and not result.rejected
    assert result.outcome.summary.startswith("No plan from 21:00 wins; this is the best attempt. Cannot finish History essay by Sun 23:30")
    why, engine = result.outcome.reasons[0], result.outcome.reasons[-1]
    assert why.startswith("cannot finish History essay by Sun 23:30: it needs 300 focused minutes and ") and "open minutes are left" in why
    assert engine.startswith("History essay: due at 23:30 with")  # the engine's own failing check is still there
    kinds = [m.kind for m in moves]
    assert "rest" not in kinds and "study" in kinds and "sleep" in kinds  # the best attempt: work on it, sleep; no filler
    assert result.totals.sleep_hours >= dayrun.cfg.win.sleep_hours
    # the planner shows it: the plan's outcome and the card carry the label
    plan, card = loop.planner.plan(at(21, 0))
    assert plan.outcome.won is False and plan.outcome.summary == result.outcome.summary
    assert "No plan from 21:00 wins" in plan.text and "No plan from 21:00 wins" in card.reason


def test_from_here_mid_afternoon_keeps_the_next_fixed_event_reachable_and_fits_dinner(busy_items, places, tmp_path):
    loop = make(busy_items, places, tmp_path, habits=FakeHabits(sleep=4.0, meals=1))
    p = loop.planner
    p.on_plan_proposed(p.plan(at(8, 16))[0])
    replan = p.replan(at(15, 0), "apartment.desk", "At Desk, you asked for a new plan.", asked=True)[0]
    judging = next(m for m in replan.moves if m.fixed and m.id.startswith(JUDGING))
    assert judging.start == at(15, 31)  # reached on time: nothing chosen before it runs into the 13 minute walk
    before = [m for m in replan.moves if not m.fixed and m.start < judging.start]
    assert all(m.end <= at(15, 18) and m.place == "apartment.desk" for m in before)
    mine = [m for m in replan.moves if not m.fixed]
    dinner = next(m for m in mine if m.kind == "eat" and m.start.date() == at(15).date())
    assert at(20, 31) <= dinner.start <= at(21, 0)  # after the study session that covers the usual dinner time
    assert replan.outcome.won and replan.outcome.meals >= 2 and replan.outcome.sleep_hours >= 6
    # the rest of the day is not turned into rest: there is a meal, the night and study for the deadline
    minutes = {k: sum(m.minutes for m in mine if m.kind == k) for k in ("rest", "study", "eat", "sleep")}
    assert minutes["study"] > 0 and minutes["rest"] < minutes["study"] and minutes["sleep"] >= 6 * 60
    # and the engine agrees nothing chosen made a fixed block late (the back-to-back study session aside)
    dayrun = p.dayrun
    board = dayrun.board(at(15, 0), InitialOverride(start_place="apartment.desk"))
    _, result = choose(dayrun.cfg, dayrun.walks, board)
    assert not result.rejected and not any("judging" in c.text for c in result.changes if c.kind == "late")


def test_the_same_board_gives_the_same_plan(busy_items, places, tmp_path):
    dayrun, board = busy(busy_items, places, tmp_path)
    first = choose(dayrun.cfg, dayrun.walks, board)
    second = choose(dayrun.cfg, dayrun.walks, board)
    _, rebuilt = busy(busy_items, places, tmp_path)
    third = choose(dayrun.cfg, dayrun.walks, rebuilt)
    assert [m.model_dump() for m in first[0]] == [m.model_dump() for m in second[0]] == [m.model_dump() for m in third[0]]
    assert first[1].model_dump() == second[1].model_dump() == third[1].model_dump()


def test_study_blocks_respect_the_preferred_block_length(places, tmp_path):
    items = items_of()
    dayrun = make(items, places, tmp_path, habits=FakeHabits()).planner.dayrun
    board = dayrun.board(at(8, 10))
    _, usual = choose(dayrun.cfg, dayrun.walks, board)
    _, short = choose(dayrun.cfg, dayrun.walks, board, preferred_block_minutes=45)
    assert max(b.minutes for b in chosen(usual, "study")) > 45 and all(b.minutes <= 90 for b in chosen(usual, "study"))
    assert chosen(short, "study") and all(b.minutes <= 45 for b in chosen(short, "study"))
    assert short.outcome.won and usual.outcome.won
    # two blocks in a row have a break between them
    for a, b in zip(chosen(short), chosen(short)[1:]):
        if a.kind == b.kind == "study":
            assert b.start - a.end >= timedelta(minutes=15)


def test_a_short_night_and_yesterdays_review_bring_bedtime_forward(places, tmp_path):
    def bedtime(sleep: float, bias=frozenset()):
        dayrun = make([], places, tmp_path, habits=FakeHabits(sleep=sleep, meals=1)).planner.dayrun
        _, result = choose(dayrun.cfg, dayrun.walks, dayrun.board(at(8, 0)), bias)
        return max(chosen(result, "sleep"), key=lambda b: b.minutes).start

    assert bedtime(7.5) == at(23, 30)
    assert bedtime(4.0) == at(23, 0)
    assert bedtime(7.5, frozenset({"earlier_sleep"})) == at(23, 0)


def test_the_scripted_demo_day_wins_with_no_fewer_stars_than_before(places, tmp_path):
    """Before this chooser: not won, 0 stars, 77 points ("The plan fails. COP4533 Project Milestone 1: at risk,
    158 focused minutes left ... (risk 0.55, needs 0.50 or less)", 595 study minutes, energy down to 0).
    After: a win with 3 stars and 99 points (317 study minutes, energy never under 38)."""
    before_stars = 0
    dayrun = make([], places, tmp_path).planner.dayrun
    board = dayrun.board(scenario="demo")
    moves, result = choose(dayrun.cfg, dayrun.walks, board)
    assert result.outcome.won and result.stars >= max(before_stars, 2) and not result.rejected
    assert result.totals.meals >= 2 and result.totals.energy_min >= dayrun.cfg.stars.energy_floor
    assert result.carry.work_left[MILESTONE] == 0 and result.totals.study_minutes < 595
    assert any(m.with_friend == "Maya" for m in moves)  # the library while Maya is there
