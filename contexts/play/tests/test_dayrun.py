"""Day Run (S4, S6): the board, the simulation, commit, the reality score and the modes."""

import itertools
import json
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from contexts.agent.application.memory_port import FakeMemory
from contexts.connectors.application.hub import ConnectorHub
from contexts.connectors.domain.spec import ConnectorSpec
from contexts.memory.fixture import load as load_memory_fixture
from contexts.play.dayrun.models import Disruption, Estimate, InitialOverride, Measured, Move
from contexts.play.dayrun.review import forecast
from contexts.play.dayrun.service import DayRun, PlanNotWinning
from contexts.play.domain.places import PlaceSet
from contracts.memory import Item

TZ = ZoneInfo("America/New_York")
SUNDAY = date(2026, 10, 11)
JUDGING = "outlook:kh-judging:2026-10-11T12:00:00-04:00"
MILESTONE = "cop4533:assignment:88213"


def at(hour: int, minute: int = 0, day: int = 11) -> datetime:
    return datetime(2026, 10, day, hour, minute, tzinfo=TZ)


def never(spec):  # a connector must not be started by anything Day Run does
    raise AssertionError("Day Run started a connector")


@pytest.fixture
def places() -> PlaceSet:
    return PlaceSet.load("config/places")


@pytest.fixture
def hub() -> ConnectorHub:
    return ConnectorHub([ConnectorSpec(name="calendar", module="x", writes=["create_event"])], factory=never)


@pytest.fixture
def dayrun(places, hub, tmp_path) -> DayRun:
    return DayRun(places, hub=hub, data_dir=tmp_path, clock=lambda: at(9, 30))


@pytest.fixture
def plans(dayrun) -> dict[str, list[Move]]:
    return dayrun.scenario("demo").plans


# --- routes: the simulation's walks are the pack's shortest routes ---


def test_routes_match_a_brute_force_search_on_the_ucf_pack(places, dayrun):
    pack = places.packs["ucf"]
    minutes = {frozenset((a, b)): m for a, b, m in pack.edges}

    def brute(a: str, b: str) -> float:  # every simple path
        best = float("inf")
        others = [p for p in pack.places if p not in (a, b)]
        for n in range(len(others) + 1):
            for mid in itertools.permutations(others, n):
                path = [a, *mid, b]
                if all(frozenset(step) in minutes for step in zip(path, path[1:])):
                    best = min(best, sum(minutes[frozenset(step)] for step in zip(path, path[1:])))
        return best

    for a, b in [("library", "towers"), ("rwc", "dining_63"), ("eng2", "library"), ("cb1", "hec"), ("towers", "rwc")]:
        path, total = places.route(f"ucf.{a}", f"ucf.{b}")
        assert total == pytest.approx(brute(a, b)), (a, b)
        trip = dayrun.walks.trip(f"ucf.{a}", f"ucf.{b}")
        assert sum(leg.minutes for leg in trip.legs) == pytest.approx(total)  # the legs add up to the route
        assert trip.legs[0].from_xy and trip.places == path


def test_a_walk_crosses_into_the_apartment_and_rounds_up(dayrun):
    minutes, trip = dayrun.walks.walk("ucf.library", "apartment.desk")
    assert trip.places[0] == "ucf.library" and trip.places[-1] == "apartment.desk" and "apartment.door" in trip.places
    assert minutes == 22 and trip.minutes == pytest.approx(21.6)
    assert dayrun.walks.walk("ucf.library", "ucf.library") == (0, None)


# --- the board ---


def test_board_lays_out_fixed_blocks_deadlines_and_open_slots(dayrun):
    b = dayrun.board(scenario="demo")
    assert (b.start, b.end) == (at(9, 30), at(9, 30, day=12))
    assert [(x.title, x.place) for x in b.blocks] == [("Knight Hacks: submissions due", None),
                                                      ("Knight Hacks: judging and demos", "ucf.eng2")]
    assert [(s.id, s.start, s.end, s.from_place, s.to_place) for s in b.slots] == [
        ("s1", at(9, 30), at(11), "ucf.eng2", None), ("s2", at(11, 15), at(12), "ucf.eng2", "ucf.eng2"),
        ("s3", at(14), at(9, 30, day=12), "ucf.eng2", None)]
    (d,) = b.deadlines
    assert (d.id, d.work_left, d.in_window) == (MILESTONE, 300, False) and 0 < d.risk < 1
    assert [e.title for e in b.events] == ["Movie on the lawn"]
    # four hours of sleep and no meal, from the habit numbers
    assert (b.initial.sleep_hours, b.initial.meals, b.initial.sleep_debt, b.initial.source) == (4, 0, 4, "habits")
    assert b.initial.energy < 50

    library = next(o for o in b.slots[2].moves if o.kind == "study" and o.place == "ucf.library")
    assert library.friends == ["Maya"] and library.walk_minutes == 9 and library.route[0] == "ucf.eng2"
    lunch = next(o for o in b.slots[1].moves if o.kind == "eat" and o.place == "ucf.dining_63")
    assert (lunch.walk_minutes, lunch.back_minutes, lunch.max_minutes) == (4, 4, 37)  # walking comes out of the slot
    nap = next(o for o in b.slots[1].moves if o.kind == "sleep")
    assert not nap.feasible and "12:00 Knight Hacks: judging and demos" in nap.reason
    assert any(o.kind == "event" and o.event.startswith("campus:lawn-movie") for o in b.slots[2].moves)


def test_board_runs_on_the_canned_demo_night(places, tmp_path):
    fx = load_memory_fixture("fixtures/memory/memory.yaml")
    dayrun = DayRun(places, memory=FakeMemory(items=fx.items), data_dir=tmp_path)
    b = dayrun.board(fx.now)
    study = next(x for x in b.blocks if x.title == "COP4533 study session")
    assert (study.place, study.activity, study.deadline) == ("ucf.library", "study", MILESTONE)  # "Library, 3rd floor"
    assert b.initial.source == "defaults" and [d.id for d in b.deadlines] == [MILESTONE]
    r = dayrun.simulate([], fx.now)
    assert r.totals.focused_minutes > 60  # the fixed study session counts towards the milestone
    assert [x.purpose for x in r.routes] == ["to Knight Hacks: judging and demos", "to COP4533 study session"]


def test_start_values_come_from_habits_or_defaults(places, tmp_path):
    class Habits:
        def digest(self, day=None, now=None):
            return {"habits": [{"id": "sleep", "value": 8.0, "applies": True}, {"id": "meals", "value": 1, "applies": True}]}

    rested = DayRun(places, habits=Habits(), data_dir=tmp_path).board(at(9, 30)).initial
    default = DayRun(places, data_dir=tmp_path).board(at(9, 30)).initial
    tired = DayRun(places, data_dir=tmp_path).board(at(9, 30), InitialOverride(sleep_hours=3)).initial
    assert rested.source == "habits" and rested.sleep_debt == 0
    assert default.source == "defaults" and default.sleep_hours == 7
    assert rested.energy > default.energy > tired.energy


# --- feasibility ---


def test_a_move_that_makes_you_late_is_rejected_with_the_minutes(dayrun):
    # s2 is 11:15 to 12:00, judging is back at Engineering II: 4 minutes each way to the dining hall
    lunch = Move(slot="s2", kind="eat", place="ucf.dining_63", minutes=40)
    r = dayrun.simulate([lunch], scenario="demo")
    (rej,) = r.rejected
    assert rej.index == 0 and rej.minutes == 3
    assert rej.reason.startswith("you miss the 12:00 Knight Hacks: judging and demos by 3 minutes")
    assert r.totals.meals == 0 and not any(b.move == 0 for b in r.blocks)

    ok = dayrun.simulate([lunch.model_copy(update={"minutes": 37})], scenario="demo")
    assert ok.rejected == [] and ok.totals.meals == 1
    meal = next(b for b in ok.blocks if b.move == 0)
    assert (meal.start, meal.end) == (at(11, 19), at(11, 56))  # the walk is taken out of the slot
    assert [(x.purpose, x.depart) for x in ok.routes][-1] == ("to Knight Hacks: judging and demos", at(11, 56))


def test_unknown_slots_and_places_are_rejected_not_fatal(dayrun):
    r = dayrun.simulate([Move(slot="s9", kind="rest"), Move(slot="s1", kind="study", place="ucf.nowhere", minutes=30),
                         Move(slot="s1", kind="eat", place="ucf.library", minutes=30)], scenario="demo")
    assert [x.index for x in r.rejected] == [0, 1, 2]
    assert "no open slot" in r.rejected[0].reason and "unknown place" in r.rejected[1].reason
    assert "not a place to eat" in r.rejected[2].reason


# --- the simulation ---


def test_the_bad_plan_fails_for_the_stated_reasons(dayrun, plans):
    r = dayrun.simulate(plans["bad"], scenario="demo")
    assert not r.outcome.won and r.stars == 0
    failed = {c.id: c for c in r.outcome.checks if not c.ok}
    assert set(failed) == {MILESTONE, "sleep", "meals"}
    assert failed["sleep"].value < 6 and failed["meals"].value == 0 and failed[MILESTONE].status == "at_risk"
    assert r.outcome.reasons == [c.text for c in failed.values()]
    # nine hours at the desk on no food and no sleep buys almost nothing
    assert r.totals.study_minutes == 540 and r.totals.focused_minutes < 60
    assert r.totals.energy_min == 0
    assert [x.reason[:44] for x in r.rejected] == ["you miss the 12:00 Knight Hacks: judging and"]


def test_the_good_plan_wins_with_three_stars(dayrun, plans):
    r = dayrun.simulate(plans["good"], scenario="demo")
    assert r.outcome.won and r.stars == r.outcome.stars == 3 and r.rejected == [] and r.outcome.reasons == []
    assert r.totals.sleep_hours >= 7.5 and r.totals.meals >= 2
    assert next(c for c in r.outcome.checks if c.id == MILESTONE).status == "on_track"
    bad = dayrun.simulate(plans["bad"], scenario="demo")
    assert r.totals.study_minutes < bad.totals.study_minutes and r.totals.focused_minutes > 4 * bad.totals.focused_minutes
    session = next(b for b in r.blocks if b.kind == "study")
    assert session.with_friend == "Maya" and session.title == "Study at John C. Hitt Library with Maya"
    # legs for the map: every route's legs join up and carry coordinates on campus
    assert r.routes and all(leg.route < len(r.routes) for leg in r.legs)
    first = [leg for leg in r.legs if leg.route == 0]
    assert (first[0].a, first[-1].b) == (r.routes[0].start, r.routes[0].end) and first[0].from_xy


def test_studying_with_a_friend_who_is_there_gets_more_done(dayrun):
    alone = dayrun.simulate([Move(slot="s3", kind="study", place="ucf.library", minutes=170)], scenario="demo")
    together = dayrun.simulate([Move(slot="s3", kind="study", place="ucf.library", minutes=170, with_friend="Maya")], scenario="demo")
    elsewhere = dayrun.simulate([Move(slot="s3", kind="study", place="ucf.hec", minutes=170, with_friend="Maya")], scenario="demo")
    assert together.totals.focused_minutes > alone.totals.focused_minutes > elsewhere.totals.focused_minutes
    assert "Maya is not at Harris Engineering Center" in elsewhere.notes[0]


def test_meters_stay_in_range_and_runs_are_identical(dayrun, plans):
    for name in ("bad", "good"):
        a = dayrun.simulate(plans[name], scenario="demo")
        b = dayrun.simulate(plans[name], scenario="demo")
        assert a.model_dump_json() == b.model_dump_json()
        assert len(a.timeline) == 24 * 4 + 1 and a.timeline[0].t == a.start and a.timeline[-1].t == a.end
        assert all(y.t - x.t == timedelta(minutes=15) for x, y in zip(a.timeline, a.timeline[1:]))
        for tick in a.timeline:
            assert 0 <= tick.energy <= 100 and 0 <= tick.focus <= 100 and 0 <= tick.deadline_risk <= 1
            assert 0 <= tick.work_left[MILESTONE] <= 300
        left = [t.work_left[MILESTONE] for t in a.timeline]
        assert left == sorted(left, reverse=True)  # work never comes back


def test_sleep_restores_energy_and_a_missed_deadline_fails(places, tmp_path):
    due = Item(connector="canvas", kind="assignment", external_id="quiz", title="Quiz 5", when=at(17))
    dayrun = DayRun(places, memory=FakeMemory(items=[due]), data_dir=tmp_path)
    r = dayrun.simulate([Move(slot="s1", kind="sleep", minutes=480)], at(9, 30))
    assert r.timeline[32].energy > r.timeline[0].energy and r.totals.sleep_hours == pytest.approx(7.97, abs=0.05)
    check = next(c for c in r.outcome.checks if c.id == "quiz")
    assert check.status == "missed" and r.timeline[-1].deadline_risk == 1
    done = dayrun.simulate([Move(slot="s1", kind="study", minutes=120)], at(9, 30))
    assert next(c for c in done.outcome.checks if c.id == "quiz").status == "done"


# --- commit ---


def test_commit_proposes_one_calendar_write_and_creates_nothing(dayrun, plans, tmp_path):
    proposal, result = dayrun.commit(plans["good"], scenario="demo", now=at(9, 31))
    assert (proposal.type, proposal.kind, proposal.target) == ("agent.action.proposed", "connector_write", "calendar.create_event")
    events = proposal.params["events"]
    assert list(proposal.params) == ["events"] and len(events) == 8 == len(result.events)
    assert all(e["title"].startswith("Day Run: ") and e["start"] < e["end"] and e["location"] for e in events)
    assert events[3] == {"title": "Day Run: Study at John C. Hitt Library with Maya", "start": "2026-10-11T14:09:00-04:00",
                         "end": "2026-10-11T16:59:00-04:00", "location": "John C. Hitt Library"}
    assert proposal.ts == at(9, 31) and result.action_id == proposal.action_id and result.stars == 3
    # nothing was created: no connector was started (the factory would raise) and no invite was written
    assert not (tmp_path / "invites").exists()
    saved = json.loads((tmp_path / "dayrun" / "plans" / "2026-10-11.json").read_text())
    assert saved["action_id"] == proposal.action_id and len(saved["moves"]) == len(plans["good"])


def test_only_a_winning_plan_commits_and_only_when_the_calendar_can_write(dayrun, plans, places, tmp_path):
    with pytest.raises(PlanNotWinning) as failed:
        dayrun.commit(plans["bad"], scenario="demo")
    assert len(failed.value.result.outcome.reasons) == 3 and not (tmp_path / "dayrun").exists()
    with pytest.raises(RuntimeError, match="cannot create events"):
        DayRun(places, data_dir=tmp_path).commit(plans["good"], scenario="demo")


# --- reality score ---


def test_reality_score_names_the_biggest_gap(dayrun, tmp_path):
    plan = [Move(slot="s1", kind="eat", place="ucf.dining_63", minutes=30),
            Move(slot="s2", kind="eat", place="ucf.union", minutes=25),
            Move(slot="s3", kind="study", place="ucf.library", minutes=170, with_friend="Maya"),
            Move(slot="s3", kind="eat", place="apartment.kitchen", minutes=30),
            Move(slot="s3", kind="study", place="apartment.desk", minutes=120),
            Move(slot="s3", kind="sleep", start=at(22, 30), minutes=480)]
    dayrun.commit(plan, scenario="demo")
    with pytest.raises(LookupError):
        dayrun.review(date(2026, 10, 9))

    perfect = dayrun.review(SUNDAY, Measured(focus_minutes=120, meals=3, sleep_hours=8))
    assert perfect.score == 100 and perfect.biggest_gap is None

    r = dayrun.review(SUNDAY, Measured(focus_minutes=30, meals=3, sleep_hours=7))
    rows = {b.id: b for b in r.blocks}
    assert rows["m4"].measured_minutes == 30 and rows["m4"].followed == 0.25  # the desk block: 30 of 120 minutes
    assert rows["m2"].followed is None and "away from the box" in rows["m2"].note  # the library cannot be measured
    assert rows["m5"].followed == pytest.approx(0.88, abs=0.01)
    assert r.score == round(100 * (0.5 * 0.25 + 0.3 * 0.875 + 0.2 * 1.0))
    assert r.biggest_gap.block == "m4" and "30 of 120 planned minutes" in r.biggest_gap.text
    assert "Biggest gap: Study at Desk" in r.text and "face presence" in r.measured_by

    # a per-block measurement (where face presence will plug in) wins over the day totals
    seen = dayrun.review(SUNDAY, Measured(focus_minutes=30, meals=1, sleep_hours=7, blocks={"m2": 170, "m4": 120}))
    assert {b.id: b.followed for b in seen.blocks}["m2"] == 1 and seen.biggest_gap.kind == "eat"
    # with no argument the numbers come from the Habits digest (the scenario's, here)
    assert dayrun.review(SUNDAY, scenario="demo").measured == Measured(focus_minutes=20, meals=2, sleep_hours=6.5)


# --- replay ---


def test_replay_scores_a_new_plan_against_what_happened(dayrun):
    actual = Measured(focus_minutes=60, meals=1, sleep_hours=4.5)
    better = [Move(slot="s1", kind="eat", place="apartment.kitchen", minutes=25),
              Move(slot="s2", kind="eat", place="ucf.union", minutes=25),
              Move(slot="s3", kind="study", place="ucf.library", minutes=170, with_friend="Maya"),
              Move(slot="s3", kind="eat", place="ucf.union", minutes=30),
              Move(slot="s3", kind="study", place="ucf.library", minutes=120),
              Move(slot="s3", kind="sleep", start=at(22, 30), minutes=510)]
    r = dayrun.replay(SUNDAY, better, actual=actual, scenario="demo")
    assert r.board.start == at(7, 30) and r.board.initial.sleep_hours == 4
    # what happened, as a plan the engine can run: that much sleep, that many meals, that much desk time
    kinds = [(m.kind, m.minutes) for m in r.baseline_moves]
    assert sorted(kinds) == [("eat", 25), ("sleep", 270), ("study", 60)] and r.baseline.rejected == []
    assert (r.baseline.totals.sleep_hours, r.baseline.totals.meals, r.baseline.totals.study_minutes) == (4.5, 1, 60)
    assert not r.baseline.outcome.won and r.plan.outcome.won
    assert r.points_gained == r.plan.totals.points - r.baseline.totals.points > 0 and r.stars_gained == r.plan.stars
    assert "beats it by" in r.text

    given = dayrun.replay(SUNDAY, better, baseline=better, actual=actual, scenario="demo")
    assert given.points_gained == 0 and "matches" in given.text


# --- hard mode ---


def test_hard_mode_revalidates_and_reroutes(dayrun, plans):
    sc = dayrun.scenario("demo")
    h = dayrun.hard(plans["good"], scenario="demo", disruptions=sc.disruptions)
    assert "moves from Engineering II (Knight Hacks) to Classroom Building I and now starts at 11:45" in h.disruptions[0].text
    judging = next(b for b in h.after.blocks if b.id == JUDGING)
    assert (judging.place, judging.start) == ("ucf.cb1", at(11, 45))
    # the walk to the moved block is routed again, from where the plan now leaves you
    was = next(r for r in h.before.routes if r.purpose == "to Knight Hacks: judging and demos")
    now = next(r for r in h.after.routes if r.purpose == "to Knight Hacks: judging and demos")
    assert was.end == "ucf.eng2" and now.end == "ucf.cb1" and now.arrive == at(11, 45) and was.places != now.places
    # the closed place's meals go to the nearest open one instead, and the squeezed one is cut to fit
    kinds = [c.kind for c in h.changes]
    assert kinds.count("rerouted") == 2 and "shortened" in kinds and "route_changed" in kinds
    eats = [b.place for b in h.after.blocks if b.kind == "eat"]
    assert "ucf.union" not in eats and len(eats) == 3
    assert h.after.rejected == [] and h.after.totals.walk_minutes > h.before.totals.walk_minutes
    assert h.before.outcome.won

    strict = dayrun.hard([Move(slot="s2", kind="eat", place="ucf.union", minutes=33)], scenario="demo",
                         disruptions=[Disruption(kind="block_moved", block=JUDGING, to_place="ucf.rwc")])
    assert strict.before.rejected == [] and [c.kind for c in strict.changes][:1] == ["shortened"]  # 11 minutes to the gym, not 6


def test_hard_mode_is_seeded_and_a_cancelled_friend_costs_the_boost(dayrun, plans):
    a = dayrun.hard(plans["good"], scenario="demo", seed=7, count=3)
    b = dayrun.hard(plans["good"], scenario="demo", seed=7, count=3)
    assert a.model_dump_json() == b.model_dump_json()
    assert sorted(d.kind for d in a.disruptions) == ["block_moved", "friend_cancelled", "place_closed"]
    assert any(c.kind == "friend_gone" for c in a.changes)
    seeds = {tuple((d.kind, d.to_place, d.place, d.shift_minutes) for d in dayrun.hard(plans["good"], scenario="demo", seed=s).disruptions)
             for s in range(12)}
    assert len(seeds) > 3  # different seeds, different days

    gone = dayrun.hard(plans["good"], scenario="demo", disruptions=[Disruption(kind="friend_cancelled", friend="Maya")])
    assert gone.after.totals.focused_minutes < gone.before.totals.focused_minutes
    assert gone.disruptions[0].text == "Maya cancels: not coming after all"


# --- campaign ---


def test_campaign_carries_energy_debt_and_work_into_the_next_day(dayrun):
    sc = dayrun.scenario("demo")
    c = dayrun.campaign(sc.campaign, scenario="demo")
    one, two = c.days
    assert (one.day, two.day) == (SUNDAY, date(2026, 10, 12))
    carry = one.result.carry
    assert two.carry_in.energy == carry.energy and two.carry_in.sleep_debt == carry.sleep_debt
    assert two.result.timeline[0].work_left == carry.work_left and carry.work_left[MILESTONE] < 300
    assert two.result.timeline[0].energy == one.result.timeline[-1].energy
    assert [b.title for b in two.result.blocks if b.fixed] == ["COP4533 lecture"]
    assert c.stars == one.result.stars + two.result.stars and c.won == (one.result.outcome.won and two.result.outcome.won)
    assert two.result.carry.work_left[MILESTONE] < carry.work_left[MILESTONE]

    # a wrecked first day makes the same second day worse
    wrecked = dayrun.campaign([[Move(slot="s3", kind="study", place="apartment.desk", minutes=900)], sc.campaign[1]], scenario="demo")
    assert wrecked.days[1].carry_in.sleep_debt > two.carry_in.sleep_debt
    assert wrecked.days[1].carry_in.energy < two.carry_in.energy
    assert wrecked.days[1].result.totals.points < two.result.totals.points and not wrecked.won


# --- forecast ---


def test_forecast_scores_calibration_against_measured_minutes(dayrun):
    f = forecast([Estimate(block="m0", minutes=60), Estimate(block="m1", minutes=30), Estimate(block="m2", minutes=45)],
                 {"m0": 120, "m1": 30})
    assert [r.error for r in f.rows] == [0.5, 0.0] and f.score == 75 and f.unmeasured == ["m2"]
    assert f.bias == -0.4 and f.multiplier == pytest.approx(1.67) and "underestimate by 40%" in f.text
    assert forecast([Estimate(block="m0", minutes=50)], {"m0": 50}).score == 100
    assert forecast([Estimate(block="x", minutes=50)], {}).score == 0
    with pytest.raises(ValueError):
        dayrun.forecast([Estimate(block="m0", minutes=50)])
