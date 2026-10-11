"""Changing the plan by saying it (contexts/play/dayrun/talk.py): the grammar, on the scripted Sunday's plan that works.

The plan: breakfast 9:38 at '63 South, a rest, [submissions due 11:00], lunch 11:22 at the Student Union, [judging
12:00 to 2:00], study at the library 2:04 with Maya, dinner 4:56 at the Student Union, study at the library 5:33, a rest
at the desk 7:31, sleep 10:31 PM. Slots: s1 9:30-11:00, s2 11:15-12:00, s3 2:00 PM to 9:30 AM.
"""

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from contexts.play.dayrun.engine import simulate
from contexts.play.dayrun.models import Move
from contexts.play.dayrun.service import DayRun, Scenario
from contexts.play.dayrun.talk import edit, normalise
from contexts.play.domain.places import PlaceSet
from contracts.memory import Item

TZ = ZoneInfo("America/New_York")
NOW = datetime(2026, 10, 11, 9, 30, tzinfo=TZ)


@pytest.fixture(scope="module")
def service(tmp_path_factory) -> DayRun:
    return DayRun(PlaceSet.load("config/places"), data_dir=tmp_path_factory.mktemp("data"), clock=lambda: NOW)


@pytest.fixture(scope="module")
def good() -> list[Move]:
    return Scenario.load("fixtures/play/demo.yaml").plans["good"]


@pytest.fixture(scope="module")
def say(service, good):
    board = service.board(scenario="demo")

    def said(text: str, moves: list[Move] | None = None):
        moves = good if moves is None else moves
        out = edit(service.cfg, service.walks, board, moves, text)
        after = simulate(service.cfg, service.walks, board, out.moves)
        block = next((b for b in after.blocks if b.move == out.changed), None) if out.changed is not None else None
        return out, block

    return said


def clock(block) -> str:
    return f"{block.start:%H:%M}"


# --- the words are tidied before they are read ---


@pytest.mark.parametrize("text, tidy", [
    ("Move lunch to one.", "move lunch to 1"),
    ("push dinner back half an hour", "push dinner back 30 minutes"),
    ("study for an hour and a half", "study for 90 minutes"),
    ("sleep at half past ten", "sleep at 10:30"),
    ("Lunch at 1 p.m.", "lunch at 1 pm"),
    ("add a nap at two thirty", "add a nap at 2:30"),
    ("study a couple of hours", "study 2 hours"),
    ("dinner at 6 o'clock", "dinner at 6"),
])
def test_normalise(text, tidy):
    assert normalise(text) == tidy


# --- move | shift | push | pull ---


@pytest.mark.parametrize("text, at, understood", [
    ("move dinner to 6:30", "18:41", "Dinner now starts at 6:41 PM, after study at John C. Hitt Library."),
    ("shift dinner to 4pm", "16:56", "Dinner now starts at 4:56 PM, after study at John C. Hitt Library."),
    ("push dinner back half an hour", "17:26", "Moved dinner to 5:26 PM."),
    ("push dinner", "17:26", "Moved dinner to 5:26 PM."),
    ("delay sleep by an hour", "23:31", "Moved sleep to 11:31 PM."),
    ("sleep at 11", "23:00", "Moved sleep to 11 PM."),
    ("hey atlas, can you move sleep to 11:15 pm please", "23:15", "Moved sleep to 11:15 PM."),
    ("lunch at 11:30", None, "Moved lunch to 11:30 AM."),
])
def test_move(say, text, at, understood):
    out, block = say(text)
    assert out.ok and out.understood == understood and len(out.moves) == 8
    if at is None:  # it was moved as asked, and the box says plainly that it no longer fits
        assert block is None and out.warning == "You'd miss judging and demos by 5 minutes"
    else:
        assert clock(block) == at and out.warning is None


def test_an_hour_with_no_am_or_pm_is_the_one_nearest_the_block(say):
    assert clock(say("move sleep to 11")[1]) == "23:00"  # not 11 in the morning
    assert say("move breakfast to 10")[0].understood.startswith("Breakfast now starts at") or clock(say("move breakfast to 10")[1]) == "10:00"


def test_a_time_that_is_taken_changes_nothing_and_offers_both_sides(say, good):
    out, _ = say("move lunch to one")
    assert not out.ok and out.moves == good
    assert out.understood == "1 PM is taken by judging and demos, so lunch can't go there."
    assert [o.label for o in out.options] == ["Move lunch to 2 PM", "Move lunch to 11:20 AM"]
    assert all(len(o.moves) == 8 for o in out.options)


def test_a_fixed_block_cannot_be_moved_and_says_why(say, good):
    out, _ = say("move judging to 3")
    assert not out.ok and out.moves == good and out.understood == "judging and demos is on your calendar, so Atlas plans around it."


# --- drop | remove | skip | cancel ---


@pytest.mark.parametrize("text, gone, understood", [
    ("skip breakfast", 0, "Dropped breakfast."),
    ("drop the rest at the desk", 6, "Dropped the rest."),
    ("remove dinner", 4, "Dropped dinner."),
    ("cancel sleep", 7, "Dropped sleep."),
    ("delete lunch", 2, "Dropped lunch."),
])
def test_drop(say, good, text, gone, understood):
    out, _ = say(text)
    assert out.ok and out.understood == understood and out.changed is None
    assert out.moves == [m for i, m in enumerate(good) if i != gone]


def test_dropping_something_that_is_not_there_offers_the_nearest_drops(say, good):
    out, _ = say("drop the gym")
    assert not out.ok and out.moves == good
    assert out.understood == "I heard “drop”, but not which block. Did you mean one of these?"
    assert len(out.options) == 2 and all(o.label.startswith("Drop ") and len(o.moves) == 7 for o in out.options)


def test_two_blocks_that_fit_equally_are_not_guessed_between(say, good):
    out, _ = say("drop study")  # there are two study blocks, both at the library
    assert not out.ok and out.moves == good
    assert [o.label for o in out.options] == ["Drop study at John C. Hitt Library at 2:04 PM", "Drop study at John C. Hitt Library at 5:33 PM"]
    first, _ = say("drop the first study")
    last, _ = say("shorten the last study to an hour")
    assert first.ok and first.moves == [m for i, m in enumerate(good) if i != 3]
    assert last.ok and last.changed == 5 and last.moves[5].minutes == 60


# --- add ---


def test_add_before_and_after_a_block(say):
    out, block = say("study two hours after judging")
    assert out.ok and out.understood == "Added study at Engineering II, 2 PM for 2 h." and (clock(block), block.minutes) == ("14:00", 120)
    out, _ = say("study two hours before judging")  # there are 45 minutes before judging, and lunch is in them
    assert not out.ok and out.understood == "There's no room to add study before judging and demos."
    assert out.options and all(o.label.startswith("Add study") for o in out.options)
    out, block = say("rest for 20 minutes after dinner")
    assert out.ok and block.kind == "rest" and block.minutes == 20 and clock(block) == "17:31"


def test_add_at_a_time_and_a_place(say, good):
    out, block = say("add study at the library at 9pm for an hour")
    assert out.ok and (block.place, block.start.hour, block.minutes) == ("ucf.library", 21, 60)
    assert out.understood == f"Added study at John C. Hitt Library, {block.start.hour - 12}:{block.start:%M} PM for 1 h."
    out, _ = say("add a nap at 3pm")  # 3 PM is in the middle of the library session
    assert not out.ok and out.moves == good and out.understood == "3 PM is in the middle of study at John C. Hitt Library."
    assert [o.label for o in out.options][0].startswith("Add sleep, ")
    out, _ = say("add")
    assert not out.ok and out.understood == "I heard “add”, but not what: a meal, study, a rest or sleep?"


def test_add_into_an_empty_day(say):
    out, block = say("add lunch", [])
    assert out.ok and block.kind == "eat" and 11 <= block.start.hour < 15 and out.understood.startswith("Added lunch, ")
    out, block = say("add dinner at the union", [])
    assert out.ok and block.place == "ucf.union" and 17 <= block.start.hour < 22
    out, block = say("sleep at 11", [])
    assert out.ok and block.kind == "sleep" and clock(block) == "23:10"  # leaving at 11, ten minutes home from Engineering II
    out, block = say("study for two hours", [])
    assert out.ok and block.kind == "study" and block.minutes == 120


def test_add_an_optional_event_by_its_name(say, service):
    out, block = say("add the movie")
    # the rest before it runs to 8:31, so the movie is joined late, and the answer says when
    assert out.ok and block.title == "Movie on the lawn" and clock(block) == "20:37"
    assert out.understood == "Added Movie on the lawn; you'd get there at 8:37 PM."

    dinner = Item(connector="calendar", kind="calendar_event", external_id="nearby:friends-dinner:x", title="Friends' dinner",
                  when=NOW.replace(hour=17, minute=15), summary="until 6:15 PM · '63 South · optional · via GroupMe")
    from contexts.play.dayrun.board import build_board, initial_state
    board = build_board(service.cfg, service.walks, NOW, [dinner], initial_state(service.cfg, NOW, None))
    out = edit(service.cfg, service.walks, board, [], "add the friends' dinner")
    assert out.ok and out.understood == "Added Friends' dinner at 5:15 PM." and out.moves[0].event == "nearby:friends-dinner:x"


# --- swap, shorten, lengthen ---


def test_swap(say, good):
    out, _ = say("swap breakfast and lunch")
    assert out.ok and out.understood == "Swapped breakfast and lunch."
    assert (out.moves[0].place, out.moves[0].slot, out.moves[2].place, out.moves[2].slot) == ("ucf.union", "s1", "ucf.dining_63", "s2")
    out, _ = say("swap lunch")
    assert not out.ok and out.moves == good and out.understood == "I heard “swap”, but I need two blocks you can move."


@pytest.mark.parametrize("text, index, minutes, understood", [
    ("shorten dinner by 10 minutes", 4, 25, "Dinner is now 25 min."),
    ("shorten dinner", 4, 20, "Dinner is now 20 min."),
    ("lengthen dinner by 15 minutes", 4, 50, "Dinner is now 50 min."),
    ("extend lunch to 40 minutes", 2, 40, "Lunch is now 40 min."),
    ("make dinner 45 minutes", 4, 45, "Dinner is now 45 min."),
    ("cut breakfast to 20 min", 0, 20, "Breakfast is now 20 min."),
])
def test_shorten_and_lengthen(say, text, index, minutes, understood):
    out, _ = say(text)
    assert out.ok and out.understood == understood and out.changed == index and out.moves[index].minutes == minutes


def test_too_short_is_refused_with_a_way_out(say, good):
    out, _ = say("shorten lunch by 20 minutes")
    assert not out.ok and out.moves == good and out.understood == "lunch can't be shorter than 15 minutes."
    assert [o.label for o in out.options] == ["Drop lunch"]


# --- never a silent no-op ---


@pytest.mark.parametrize("text", ["", "   ", "banana", "what is the weather", "move", "drop", "lunch", "to 3pm", "12345"])
def test_what_cannot_be_read_changes_nothing_and_says_so(say, good, text):
    out, _ = say(text)
    assert not out.ok and out.moves == good and out.understood and out.changed is None
    assert all(o.moves != good and o.label for o in out.options)
    assert bool(out.options) != bool(out.examples)  # the nearest edits, or (nothing read at all) what can be said: never both


@pytest.mark.parametrize("text", ["", "banana", "what is the weather", "to 3pm", "12345"])
def test_when_nothing_at_all_is_read_it_says_what_can_be_said_about_this_plan_and_offers_no_edit(say, good, text):
    out, _ = say(text)
    assert not out.ok and out.moves == good and out.options == []
    assert out.understood == ("I couldn't tell what to change." if text else "I didn't catch anything to change.")
    # one to move, one to drop, one to add, in the names of this plan's own blocks
    assert out.examples == ["move breakfast to 10 AM", "drop the second study block", "add Movie on the lawn"]
    moved, block = say(out.examples[0])  # and each of them is read, and does what it says
    assert moved.ok and moved.understood == "Moved breakfast to 10 AM." and clock(block) == "10:00"
    dropped, _ = say(out.examples[1])
    assert dropped.ok and dropped.moves == [m for i, m in enumerate(good) if i != 5]
    added, block = say(out.examples[2])
    assert added.ok and block.title == "Movie on the lawn"


def test_a_part_read_keeps_its_nearest_edits_and_gets_no_examples(say):
    for text in ("move", "drop", "lunch", "drop the gym"):  # a verb or a block was read: the offer is about that
        out, _ = say(text)
        assert not out.ok and out.options and out.examples == [], text


def test_the_examples_follow_the_plan_they_are_about(say, service):
    bad = Scenario.load("fixtures/play/demo.yaml").plans["bad"]
    for moves in (bad, []):
        out, _ = say("banana", moves)
        assert not out.ok and out.options == [] and 2 <= len(out.examples) <= 3
        for phrase in out.examples:
            tried, _ = say(phrase, moves)
            assert tried.ok, phrase
    assert say("banana", [])[0].examples == ["add lunch", "add study for 2 hours", "add Movie on the lawn"]
    assert say("banana", bad)[0].examples[0] == "move study at Desk to 8 PM"


def test_every_answer_is_a_short_sentence(say):
    for text in ("move dinner to 6:30", "drop the gym", "move lunch to one", "study two hours before judging", "add a nap at 3pm", "banana"):
        out, _ = say(text)
        assert len(out.understood.split()) <= 14, out.understood
        assert all(len(o.label.split()) <= 10 for o in out.options)
        assert all(len(e.split()) <= 8 for e in out.examples)
