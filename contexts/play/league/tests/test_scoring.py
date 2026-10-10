"""The scoring rule, the worst/best rules and the team waste line: pure functions."""

from datetime import date, timedelta

from contexts.play.league.config import LeagueConfig, TargetRule, load_config
from contexts.play.league.domain import scoring
from contexts.play.league.domain.scoring import Count, DaySummary, PlanStars
from contexts.play.league.tests.helpers import MONDAY, NAMES, SCHED, at
from contracts.play import Stake


def sleep_week(member: str, met_nights: int, nights: int = 7) -> list[DaySummary]:
    return [DaySummary(member, "sleep", MONDAY + timedelta(days=i), i < met_nights, 7.0 if i < met_nights else 4.0)
            for i in range(nights)]


def test_a_week_is_the_weighted_share_of_judged_target_days_hit():
    cfg = LeagueConfig(targets={"sleep": TargetRule(weight=2, label="sleep", unit="nights"), "meals": TargetRule(label="meals")})
    habits = sleep_week("hemdutt", 6) + [DaySummary("hemdutt", "meals", MONDAY + timedelta(days=i), i % 2 == 0, 2) for i in range(4)]
    [week] = scoring.score_week(cfg, MONDAY, {"hemdutt": "Hemdutt"}, habits, [], [], SCHED.local_day)
    # sleep 6/7 at weight 2, meals 2/4 at weight 1 -> (2*0.857 + 0.5) / 3
    assert week.score == round(100 * (2 * 6 / 7 + 2 / 4) / 3, 1) == 73.8
    assert (week.met, week.judged) == (8, 11)
    assert week.reasons() == ["missed meals 2 of 4 days", "missed sleep 1 of 7 nights"]
    assert week.headline() == "2 of 4 meals targets"


def test_days_without_a_summary_are_not_judged_and_a_member_with_nothing_has_no_score():
    cfg = load_config(None)
    habits = sleep_week("maya", 2, nights=3)
    weeks = scoring.score_week(cfg, MONDAY, NAMES, habits, [], [], SCHED.local_day)
    by = {w.member_id: w for w in weeks}
    assert by["maya"].judged == 3 and by["maya"].score == 66.7
    assert by["hemdutt"].score is None and by["dom"].score is None
    assert [w.member_id for w in weeks] == ["maya", "dom", "hemdutt"]  # scored first, then the rest by id


def test_worst_is_the_lowest_score_best_the_highest_and_nobody_without_data_can_be_either():
    cfg = load_config(None)
    habits = sleep_week("hemdutt", 6) + sleep_week("maya", 2) + sleep_week("dom", 4)
    weeks = scoring.score_week(cfg, MONDAY, {**NAMES, "ilan": "Ilan"}, habits, [], [], SCHED.local_day)
    worst, best = scoring.worst_and_best(weeks)
    assert (worst.member_id, best.member_id) == ("maya", "hemdutt")
    assert worst.reasons() == ["missed sleep 5 of 7 nights"] and worst.headline() == "2 of 7 sleep targets"
    assert scoring.summary_text(worst, best, Stake(kind="pays_for_food")) == (
        "Maya had the worst week: 2 of 7 sleep targets. Dinner is on Maya. Hemdutt picks the place.")
    assert scoring.summary_text(worst, best, Stake(kind="picks_the_event")) == (
        "Maya had the worst week: 2 of 7 sleep targets. Hemdutt had the best week (6 of 7 sleep targets) and picks the next outing.")
    assert scoring.summary_text(worst, best, Stake(kind="custom", text="the loser does the dishes")) == (
        "Maya had the worst week: 2 of 7 sleep targets. the loser does the dishes.")


def test_ties_break_on_more_target_days_hit_then_on_the_member_id_so_a_replay_picks_the_same_person():
    cfg = load_config(None)
    habits = sleep_week("maya", 2, nights=4) + sleep_week("dom", 2, nights=4)  # both 50%
    worst, best = scoring.worst_and_best(scoring.score_week(cfg, MONDAY, NAMES, habits, [], [], SCHED.local_day))
    assert (best.member_id, worst.member_id) == ("dom", "maya")
    habits = sleep_week("maya", 2, nights=4) + sleep_week("dom", 3, nights=6)  # both 50%, dom hit more
    worst, best = scoring.worst_and_best(scoring.score_week(cfg, MONDAY, NAMES, habits, [], [], SCHED.local_day))
    assert (best.member_id, worst.member_id) == ("dom", "maya")


def test_one_member_with_data_is_the_best_and_there_is_no_worst():
    weeks = scoring.score_week(load_config(None), MONDAY, NAMES, sleep_week("hemdutt", 7), [], [], SCHED.local_day)
    worst, best = scoring.worst_and_best(weeks)
    assert worst is None and best.member_id == "hemdutt"
    assert scoring.summary_text(worst, best, Stake()) == "Only Hemdutt had a judged week (7 of 7 sleep targets). Nobody owes anything."
    assert scoring.summary_text(None, None, Stake()) == "Nothing was judged this week, so nobody owes anything."


def test_day_run_stars_and_the_weeks_recycling_count_as_targets_too():
    cfg = load_config(None)
    plans = [PlanStars("hemdutt", MONDAY, 3), PlanStars("hemdutt", MONDAY + timedelta(days=1), 1)]
    counts = [Count("hemdutt", at(MONDAY - timedelta(days=1)), 10, 6, 3, 1),  # before the week: the baseline
              Count("hemdutt", at(MONDAY + timedelta(days=3)), 20, 16, 12, 2)]  # 9 of 10 recyclables this week
    [week] = scoring.score_week(cfg, MONDAY, {"hemdutt": "Hemdutt"}, [], counts, plans, SCHED.local_day)
    assert week.targets["dayrun_stars"].met == 1 and week.targets["dayrun_stars"].judged == 2
    assert week.targets["recycling"].met == 1 and week.targets["recycling"].detail == "recycled 90% of what could be (target 80%)"
    assert week.reasons()[0] == "missed Day Run stars 1 of 2 plans"


def test_the_team_waste_line_adds_everyones_bins_and_names_the_contamination_kindly():
    cfg = load_config(None)
    counts = [Count("hemdutt", at(MONDAY + timedelta(days=2)), 9, 6, 4, 1), Count("maya", at(MONDAY + timedelta(days=2)), 12, 7, 4, 2),
              Count("dom", at(MONDAY - timedelta(days=3)), 50, 40, 40, 0)]  # dom counted nothing this week
    team = scoring.team_waste(cfg, NAMES, counts, MONDAY, SCHED.local_day)
    assert [r["member_id"] for r in team.members] == ["hemdutt", "maya"]
    assert (team.recycled, team.recyclable, team.rate) == (8, 13, 0.615)
    assert team.line == "team: 62% diverted, target 80%"
    assert team.contamination_of_the_week.startswith("Maya's bin had the most mix-ups this week (2 items")
    empty = scoring.team_waste(cfg, NAMES, [], MONDAY, SCHED.local_day)
    assert empty.rate is None and empty.line == "team: nothing counted yet this week, target 80%" and empty.contamination_of_the_week is None


def test_the_week_starts_on_the_local_monday():
    assert scoring.week_of(date(2026, 10, 11)) == MONDAY == scoring.week_of(MONDAY)
    assert scoring.week_of(date(2026, 10, 12)) == MONDAY + timedelta(days=7)
    assert scoring.week_end(MONDAY) == date(2026, 10, 11)


def test_the_shipped_config_loads_with_the_owners_targets_and_presets():
    cfg = load_config("config/league.yaml")
    assert set(cfg.targets) == {"sleep", "meals", "focus", "energy_drinks", "leave_ready", "sort_right", "dayrun_stars", "recycling"}
    assert cfg.targets["sleep"].weight == 2 and cfg.stake.kind == "pays_for_food" and cfg.team_diversion_target == 0.8
    assert cfg.presets["no_energy_drinks"].habit == "energy_drinks" and cfg.close_grace_hours == 3
    assert load_config("config/does-not-exist.yaml").targets == cfg.targets
