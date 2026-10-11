"""The arbiter's rules (contexts/day/domain/arbiter.py): order, the global gap, merging, suppression, the push key."""

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from contexts.day.domain import arbiter as rule
from contexts.day.domain import narration as words
from contexts.day.domain.arbiter import Ledger, Line, Rules, Scene, decide

TZ = ZoneInfo("America/New_York")
T0 = datetime(2026, 10, 12, 14, 0, tzinfo=TZ)


def s(seconds: float) -> datetime:
    return T0 + timedelta(seconds=seconds)


def said(moment: str, at: datetime, urgent: bool = False, plan_id: str | None = None, event_id: str = "x") -> Ledger:
    ledger = Ledger()
    ledger.note_said(moment, at, urgent=urgent, plan_id=plan_id, event_id=event_id)
    return ledger


ORDER = ["approval", "checkin", "start", "suggest", "coach", "evening", "nudge"]


@pytest.mark.parametrize("high,low", [(ORDER[i], ORDER[j]) for i in range(len(ORDER)) for j in range(i + 1, len(ORDER))])
def test_each_pair_is_ordered_highest_first(high, low):
    a, b = Line(low, T0), Line(high, T0)
    assert [x.moment for x in rule.order([a, b])] == [high, low]


def test_urgent_outranks_everything_and_morning_and_replan_rank_with_approvals():
    lines = [Line("approval", T0), Line("now", T0, urgent=True), Line("coach", T0)]
    assert [x.moment for x in rule.order(lines)] == ["now", "approval", "coach"]
    assert Line("morning", T0).rank == Line("replan", T0).rank == Line("approval", T0).rank


def test_the_global_gap_holds_a_second_line_and_frees_the_slot_after_it():
    ledger = said("coach", s(0))
    assert decide(Line("start", s(30)), Scene(), ledger, s(30)).action == "hold"
    assert decide(Line("start", s(30)), Scene(), ledger, s(89)).action == "hold"
    assert decide(Line("start", s(30)), Scene(), ledger, s(90)).action == "say"
    assert decide(Line("start", s(30)), Scene(), ledger, s(90), Rules(gap_s=120)).action == "hold"


def test_urgent_breaks_the_gap_and_an_alert_breaks_anything_but_another_urgent_line():
    assert decide(Line("now", s(10), urgent=True), Scene(), said("start", s(0)), s(10)).action == "say"
    assert decide(Line("alert", s(10)), Scene(), said("start", s(0)), s(10)).action == "say"
    assert decide(Line("alert", s(10)), Scene(), said("now", s(0), urgent=True), s(10)).action == "hold"


def test_a_plans_own_card_is_said_inside_the_plans_narration():
    ledger = said("morning", s(0), plan_id="p1", event_id="day-narrated-plan")
    v = decide(Line("approval", s(0), plan_id="p1"), Scene(), ledger, s(0))
    assert (v.action, v.into) == ("companion", "day-narrated-plan")
    assert decide(Line("approval", s(0), plan_id="p2"), Scene(), ledger, s(0)).action == "hold"  # another plan's card waits


def test_a_stale_line_is_dropped_but_a_patient_one_waits():
    ledger = said("coach", s(0))
    assert decide(Line("start", s(0)), Scene(), ledger, s(121)).action == "drop"
    assert decide(Line("checkin", s(0)), Scene(), ledger, s(500)).action == "say"


def test_no_block_start_for_an_event_that_had_a_leave_now_line_in_the_last_ten_minutes():
    ledger = Ledger()
    ledger.note_urgent(["dinner"], s(0))
    assert decide(Line("start", s(60), about=frozenset({"dinner"})), Scene(), ledger, s(60)).action == "drop"
    assert decide(Line("start", s(60), about=frozenset({"study"})), Scene(), ledger, s(60)).action == "say"
    assert decide(Line("start", s(600), about=frozenset({"dinner"})), Scene(), ledger, s(600)).action == "say"


def test_no_suggestion_while_a_card_is_open_or_soon_after_a_check_in():
    ledger = Ledger()
    ledger.note_card("card-1", s(300))
    assert decide(Line("suggest", s(60)), Scene(), ledger, s(60)).reason == "an approval card is open"
    ledger.note_card_closed("card-1")
    assert decide(Line("suggest", s(60)), Scene(), ledger, s(60)).action == "say"
    ledger.note_checkin("ci-1", s(0))
    ledger.note_checkin_answered("ci-1")
    assert decide(Line("suggest", s(14 * 60)), Scene(), ledger, s(14 * 60)).reason == "a check-in was said in the last 15 minutes"
    assert decide(Line("suggest", s(15 * 60)), Scene(), ledger, s(15 * 60)).action == "say"
    ledger.note_checkin("ci-2", s(15 * 60))  # unanswered: open for an hour
    assert decide(Line("suggest", s(40 * 60)), Scene(), ledger, s(40 * 60)).reason == "a check-in card is open"
    ledger.note_card("expired", s(0))
    ledger.note_checkin_answered("ci-2")
    assert decide(Line("suggest", s(40 * 60)), Scene(), ledger, s(40 * 60)).action == "say"  # an expired card is not open


def test_no_coach_note_or_block_start_during_a_fixed_event():
    assert decide(Line("coach", T0), Scene(fixed_running=True), Ledger(), T0).action == "drop"
    assert decide(Line("start", T0), Scene(fixed_running=True), Ledger(), T0).action == "drop"
    assert decide(Line("checkin", T0), Scene(fixed_running=True), Ledger(), T0).action == "say"


def test_nothing_non_urgent_while_an_answer_is_said_and_nothing_but_urgent_at_night():
    assert decide(Line("start", T0), Scene(answering=True), Ledger(), T0).action == "hold"
    assert decide(Line("start", T0), Scene(answering=True), Ledger(), s(121)).action == "drop"
    assert decide(Line("start", T0), Scene(awake=False), Ledger(), T0).action == "drop"
    assert decide(Line("now", T0, urgent=True), Scene(awake=False), Ledger(), T0).action == "say"


def test_only_a_short_form_is_merged():
    assert rule.mergeable(Line("start", T0), Line("suggest", T0))
    assert rule.mergeable(Line("checkin", T0), Line("start", T0))
    assert not rule.mergeable(Line("start", T0), Line("coach", T0))
    assert not rule.mergeable(Line("approval", T0), Line("suggest", T0))
    assert not rule.mergeable(Line("now", T0, urgent=True), Line("suggest", T0))


def test_the_push_key_is_day_moment_and_ref():
    assert rule.key(date(2026, 10, 12), "morning", "e1") == "2026-10-12:morning:e1"
    assert rule.key(date(2026, 10, 12), "morning", "e1") != rule.key(date(2026, 10, 12), "replan", "e1")


START = {"moment": "start", "block": "lunch", "kind": "eat", "at": "12:00", "place": "Student Union", "walk_min": 7}
SUGGEST = {"moment": "suggest", "title": "Friends' dinner", "who": ["Riley"], "at": "17:15", "until": "18:15",
           "place": "'63 South", "walk_min": 7, "fits": "yes"}


def test_two_lines_due_together_come_out_as_one_sentence_from_both_facts():
    merged = words.render(dict(START, also=SUGGEST))
    assert merged.spoken == "Time for lunch at Student Union; it's a seven-minute walk; and Riley's dinner is at '63 South at quarter past five if you want it."
    assert merged.spoken.count(".") == 1
    assert merged.display.startswith(words.render(START).display) and merged.display.endswith(words.render(SUGGEST).display)
    assert words.check(merged.spoken, dict(START, also=SUGGEST), merged.spoken) == []


def test_a_merge_falls_back_to_the_first_line_when_it_ends_with_a_question_or_the_second_has_no_short_form():
    coach = {"moment": "coach", "title": "Study", "planned_min": 25, "desk_min": 25, "breaks": 0}
    assert words.render(dict(START, also=coach)) == words.render(START)
