"""Today's Plan in words: every sentence about a number is held to the fields the engine used for that number."""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from contexts.play.dayrun import fixes, words
from contexts.play.dayrun.board import build_board, initial_state
from contexts.play.dayrun.config import DayRunConfig
from contexts.play.dayrun.engine import simulate
from contexts.play.dayrun.models import InitialOverride, Move
from contexts.play.dayrun.routes import Walks
from contexts.play.dayrun.service import DayRun, Scenario
from contexts.play.dayrun_api import dayrun_router
from contexts.play.domain.places import PlaceSet
from contracts.memory import Item

TZ = ZoneInfo("America/New_York")
NOW = datetime(2026, 10, 11, 9, 30, tzinfo=TZ)
MILESTONE = "cop4533:assignment:88213"


@pytest.fixture(scope="module")
def service(tmp_path_factory) -> DayRun:
    return DayRun(PlaceSet.load("config/places"), data_dir=tmp_path_factory.mktemp("data"), clock=lambda: NOW)


@pytest.fixture(scope="module")
def plans() -> dict:
    return Scenario.load("fixtures/play/demo.yaml").plans


def run(service, moves, initial=None):
    board = service.board(initial=initial, scenario="demo")
    return board, simulate(service.cfg, service.walks, board, moves)


def test_times_and_lengths_read_as_a_person_writes_them():
    assert [words.clock12(datetime(2026, 10, 11, h, m, tzinfo=TZ)) for h, m in ((9, 5), (14, 0), (12, 0), (0, 0), (23, 59))] == [
        "9:05 AM", "2 PM", "noon", "midnight", "11:59 PM"]
    assert [words.dur(n) for n in (40, 60, 150, 281.4)] == ["40 min", "1 h", "2 h 30 min", "4 h 41 min"]
    assert [words.hours(h) for h in (1, 5.33, 8.0)] == ["1 hour", "5.3 hours", "8 hours"]
    assert [words.span(h) for h in (5, 38, 48)] == ["5 h", "1 day 14 h", "2 days"]
    assert words.short("Knight Hacks: judging and demos") == "judging and demos" and words.short("COP4533 lecture") == "COP4533 lecture"
    assert words.assignment("Project - Milestone1 Fall2026 [COP4533]") == "Project Milestone 1"
    assert [words.assignment(t) for t in ("Quiz10", "HW3 [COP4533]", "Lab 2", "COP4533")] == ["Quiz 10", "HW3", "Lab 2", "COP4533"]
    assert words.assignment("COP4533 Project Milestone 1") == "Project Milestone 1"
    assert [words.headline(n) for n in (0, 1, 2, 3)] == ["This plan works", "One thing to fix", "Two things to fix", "Three things to fix"]


def test_work_left_says_it_is_a_guess_and_which_kind(service):
    board, _ = run(service, [])
    d = board.deadlines[0]
    assert (d.work_left, d.estimate_from, d.estimate_word, d.estimate_minutes) == (300, "kind", "project", 300)  # config/dayrun.yaml work_estimates
    said = words.why_work(d)
    assert said.startswith(f"About {words.dur(d.work_left)}: Atlas's starting guess for a {d.estimate_word}") and "not measured" in said

    cfg = service.cfg
    plain = build_board(cfg, service.walks, NOW, [Item(connector="canvas", kind="assignment", external_id="x", title="Reading response",
                                                       when=NOW + timedelta(hours=30))], initial_state(cfg, NOW, None))
    other = plain.deadlines[0]
    assert (other.work_left, other.estimate_from, other.estimate_word) == (cfg.default_work_minutes, "default", None)
    assert words.why_work(other).startswith(f"About {words.dur(cfg.default_work_minutes)}: Atlas's starting guess for any assignment")


def test_your_own_estimate_is_called_yours_and_keeps_the_guess_beside_it(service):
    board, _ = run(service, [], InitialOverride(work_left={MILESTONE: 180}))
    d = board.deadlines[0]
    assert (d.work_left, d.estimate_from, d.estimate_minutes) == (180, "yours", 300)
    assert words.why_work(d) == "3 h of work: your own estimate. Atlas guessed 5 h."


def test_at_risk_and_on_track_sentences_carry_the_work_done_the_work_left_and_the_rate(service, plans):
    board, bad = run(service, plans["bad"])
    c = next(c for c in bad.outcome.checks if c.id == MILESTONE)
    assert c.status == "at_risk" and c.done_minutes + c.left_minutes == pytest.approx(300, abs=0.2) and c.due == board.deadlines[0].due
    assert c.hours_left == pytest.approx((c.due - board.end).total_seconds() / 3600, abs=0.01)
    said = words.why_risk(c, board)
    assert said == (f"This plan does {words.dur(c.done_minutes)}, leaving {words.dur(c.left_minutes)}. "
                    f"At {words.dur(board.rules.capacity_minutes_per_day)} of real work a day, too tight for Tuesday 11:59 PM.")
    assert board.rules.capacity_minutes_per_day == service.cfg.capacity_minutes_per_day == 180
    # the rule behind "too tight": the work left against the capacity before it is due
    assert c.value == pytest.approx(min(1, c.left_minutes / (c.hours_left * 180 / 24)), abs=0.002) and c.value > c.target

    _, good = run(service, plans["good"])
    ok = next(c for c in good.outcome.checks if c.id == MILESTONE)
    assert ok.status == "on_track"
    assert words.why_risk(ok, board) == (f"This plan does {words.dur(ok.done_minutes)}. The {words.dur(ok.left_minutes)} left fits "
                                         "before Tuesday 11:59 PM at 3 h a day.")
    _, empty = run(service, [])
    none = next(c for c in empty.outcome.checks if c.id == MILESTONE)
    assert none.done_minutes == 0 and words.why_risk(none, board).startswith("This plan puts no work into it, leaving 5 h.")


def test_sleep_and_meals_say_where_last_night_and_the_meals_came_from(service, plans):
    board, good = run(service, plans["good"])
    assert board.initial.source == "scenario" and board.initial.sleep_hours == 4
    assert words.why_sleep(good, board) == (f"{words.hours(good.totals.sleep_hours)} in this plan; Atlas asks for 6. "
                                            "Last night: 4 hours, as this scripted day says.")
    assert words.why_meals(good, board) == f"{good.totals.meals} meals in this plan, none seen before it starts. Atlas asks for 2."

    cfg, w = service.cfg, service.walks
    for rows, source, expected in (
            (None, None, "Last night wasn't measured, so Atlas assumed 7 hours."),  # config: start.default_sleep_hours
            ([{"id": "sleep", "value": 5.5, "source": "watch"}], "watch", "Last night: 5.5 hours, from your watch."),
            ([{"id": "sleep", "value": 6.5, "source": "camera"}], "camera", "Last night: 6.5 hours, from the desk camera.")):
        state = initial_state(cfg, NOW, rows)
        assert state.sleep_from == source
        live = build_board(cfg, w, NOW, [], state)
        assert words.why_sleep(simulate(cfg, w, live, []), live).endswith(expected)

    seen = build_board(cfg, w, NOW, [], initial_state(cfg, NOW, [{"id": "meals", "value": 1}]))
    one = simulate(cfg, w, seen, [])
    assert (one.totals.meals, one.totals.meals_before) == (1, 1)
    assert words.why_meals(one, seen) == "1 meal today: 1 already seen at your desk, 0 in this plan. Atlas asks for 2."


def test_a_walks_minutes_are_its_metres_at_the_packs_pace(service, plans):
    board, good = run(service, plans["good"])
    pace = board.rules.walk_meters_per_minute
    assert pace == pytest.approx(1.35 * 60)  # walknet's default pace, the one the routes are timed at
    outdoor = next(r for r in good.routes if r.meters)
    assert outdoor.minutes == pytest.approx(outdoor.meters / pace, abs=0.5)  # steps along the way count a little slower
    assert words.why_walk(outdoor, board) == (f"{round(outdoor.minutes)} minutes: about {round(outdoor.meters / 10) * 10} metres "
                                              "along the mapped footpaths, at an easy walking pace.")
    indoor = next(r for r in good.routes if not r.meters)
    assert words.why_walk(indoor, board).endswith("the set walking time between these two places.")


def test_points_add_up_from_their_four_parts_and_stars_say_what_they_took(service, plans):
    board, good = run(service, plans["good"])
    parts = good.totals.points_parts
    assert set(parts) == {"deadlines", "sleep", "meals", "energy"} and abs(sum(parts.values()) - good.totals.points) <= 2
    assert words.why_points(good) == (f"{good.totals.points} of 100: deadlines {parts['deadlines']} of 40, sleep {parts['sleep']} of 25, "
                                      f"meals {parts['meals']} of 15, energy left at the end {parts['energy']} of 20.")
    assert good.stars == 3 and words.why_stars(good).startswith("Three stars:")
    _, bad = run(service, plans["bad"])
    assert words.why_stars(bad).startswith("No stars yet")


def test_what_needs_fixing_is_plain_and_each_fix_is_tried_first(service, plans):
    board, bad = run(service, plans["bad"])
    found = fixes.suggest(service.cfg, service.walks, board, plans["bad"], bad)
    by = {f.id: f for f in found}
    assert set(by) == {"move:1", MILESTONE, "sleep", "meals"}
    late = by["move:1"]
    assert late.problem == f"You'd miss judging and demos by {bad.rejected[0].minutes} minutes" and bad.rejected[0].minutes == 11
    assert late.label == "Shorten the meal at '63 South to 25 min" and late.mends and late.moves[late.changed].minutes == 25
    assert not simulate(service.cfg, service.walks, board, late.moves).rejected  # the edit really makes it fit
    sleep = next(c for c in bad.outcome.checks if c.id == "sleep")
    assert by["sleep"].problem == f"{words.hours(sleep.value)} of sleep; you need 6" and by["sleep"].mends
    after = simulate(service.cfg, service.walks, board, by["sleep"].moves)
    assert next(c for c in after.outcome.checks if c.id == "sleep").ok
    assert by["meals"].problem == "No meals today; you need 2" and by["meals"].label.startswith("Add a meal at ")
    assert by[MILESTONE].problem.startswith("Project Milestone 1 is at risk: ")
    for line in [f.problem for f in found] + [f.label for f in found if f.label]:
        assert len(line.split()) <= 12, line

    _, good = run(service, plans["good"])
    assert fixes.suggest(service.cfg, service.walks, board, plans["good"], good) == []


def test_a_problem_with_no_edit_yet_names_the_one_to_fix_first(service, plans):
    board, bad = run(service, plans["bad"])
    found = fixes.suggest(service.cfg, service.walks, board, plans["bad"], bad)
    by = {f.id: f for f in found}
    stuck = by[MILESTONE]  # the desk session runs past bedtime: there is no room for more work until the night is settled
    assert stuck.label is None and stuck.moves is None
    assert (stuck.after, stuck.hint) == ("sleep", "Fix your sleep first")
    assert all(f.after is None and f.hint is None for f in found if f.label)  # only a row with no button carries the line
    # it is true, not guessed: once the sleep is fixed, the milestone has an edit of its own
    then = by["sleep"].moves
    again = {f.id: f for f in fixes.suggest(service.cfg, service.walks, board, then, simulate(service.cfg, service.walks, board, then))}
    assert again[MILESTONE].label and again[MILESTONE].hint is None
    # and fixing the row above it (the meal that runs into judging) would not have done it
    late = by["move:1"].moves
    still = {f.id: f for f in fixes.suggest(service.cfg, service.walks, board, late, simulate(service.cfg, service.walks, board, late))}
    assert still[MILESTONE].label is None and still[MILESTONE].after == "sleep"
    # the row right above needs no name
    assert words.fix_first("sleep", board, above=True) == "Fix the one above first"
    assert words.fix_first(MILESTONE, board, above=False) == "Fix Project Milestone 1 first"
    assert words.fix_first("move:1", board, above=False) == "Fix what doesn't fit first"


def test_fixing_one_thing_at_a_time_ends_in_a_plan_that_works(service):
    board = service.board(scenario="demo")
    moves: list[Move] = []
    for _ in range(12):
        result = simulate(service.cfg, service.walks, board, moves)
        found = [f for f in fixes.suggest(service.cfg, service.walks, board, moves, result) if f.moves is not None]
        if not found:
            break
        moves = (next((f for f in found if f.mends), None) or found[0]).moves
    assert result.outcome.won and not result.rejected


def test_every_sentence_is_one_or_two_short_lines(service, plans):
    for name in ("bad", "good"):
        board, result = run(service, plans[name])
        out = fixes.explain(service.cfg, service.walks, board, plans[name], result)
        said = [out["sleep"], out["meals"], out["points"], out["stars"], *out["walks"].values()]
        said += [d[k] for d in out["deadlines"].values() for k in ("work", "risk")]
        for text in said:
            assert len(text.split()) <= 26 and not any(bad in text for bad in ("risk 0", "0.5", "scenario=", "s1", MILESTONE)), text


def test_near_you_events_are_offered_in_their_gap_and_never_placed(service):
    cfg, w = service.cfg, service.walks
    dinner = Item(connector="calendar", kind="calendar_event", external_id="nearby:friends-dinner:2026-10-11T17:15:00-04:00",
                  title="Friends' dinner", when=NOW.replace(hour=17, minute=15), summary="until 6:15 PM · '63 South · with Riley, Sam · optional · via GroupMe")
    lecture = Item(connector="calendar", kind="calendar_event", external_id="outlook:lecture", title="COP4533 lecture",
                   when=NOW.replace(hour=12, minute=0), summary="until 12:50 PM · HEC 101")
    board = build_board(cfg, w, NOW, [dinner, lecture], initial_state(cfg, NOW, None))
    assert [b.id for b in board.blocks] == ["outlook:lecture"] and board.events == []  # still never a block or a listed move
    assert all(o.event is None for s in board.slots for o in s.moves)
    (s,) = board.suggestions
    slot = next(x for x in board.slots if x.id == s.slot)
    assert s.title == "Friends' dinner" and s.note == "with Riley, Sam · optional · via GroupMe" and s.place == "ucf.dining_63"
    assert s.walk_minutes == w.walk(slot.from_place, s.place)[0] and s.leave_by == s.start - timedelta(minutes=s.walk_minutes)

    went = simulate(cfg, w, board, [Move(slot=s.slot, kind="event", event=s.id)])
    block = next(b for b in went.blocks if b.kind == "event")
    assert (block.title, block.start, block.end) == ("Friends' dinner", s.start, s.end) and not went.rejected
    walk = next(r for r in went.routes if r.move == 0)
    assert walk.depart == s.leave_by and walk.end == "ucf.dining_63"


def test_explain_and_edit_over_http_and_the_old_answers_are_untouched(service, plans):
    app = FastAPI()
    app.include_router(dayrun_router(service))
    client = TestClient(app)
    moves = [m.model_dump(mode="json", exclude_none=True) for m in plans["bad"]]
    out = client.post("/api/dayrun/explain", json={"scenario": "demo", "moves": moves}).json()
    assert out["headline"] == "Four things to fix" and [f["id"] for f in out["fixes"]][0] == "move:1"
    assert set(out) == {"headline", "fixes", "deadlines", "sleep", "meals", "points", "stars", "walks"}
    assert out["deadlines"][MILESTONE]["title"] == "Project Milestone 1" and out["deadlines"][MILESTONE]["due"] == "Tuesday 11:59 PM"
    mine = client.post("/api/dayrun/explain", json={"scenario": "demo", "moves": moves, "initial": {"work_left": {MILESTONE: 60}}}).json()
    assert mine["deadlines"][MILESTONE]["work"].startswith("1 h of work: your own estimate.")

    said = client.post("/api/dayrun/edit", json={"scenario": "demo", "moves": moves, "text": "drop the lunch at 63 south"})
    assert said.status_code == 200 and said.json()["ok"] and len(said.json()["moves"]) == len(moves) - 1
    assert client.post("/api/dayrun/edit", json={"scenario": "demo", "moves": moves, "text": "x" * 301}).status_code == 422
    sim = client.post("/api/dayrun/simulate", json={"scenario": "demo", "moves": moves}).json()
    assert "fixes" not in sim and len(sim["outcome"]["reasons"]) == 3  # the simulate answer keeps its shape
