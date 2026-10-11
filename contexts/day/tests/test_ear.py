# OWNER'S RULE (Hemdutt): Atlas says things like a person and never reads a record out: no time ranges, ids, tags or brackets.
# These pin the wording every fixed answer is built from (contexts/day/domain/ear.py). Change them only with the owner.
import re
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from contexts.day.domain import ear

TZ = ZoneInfo("America/New_York")
NOW = datetime(2026, 10, 11, 9, 0, tzinfo=TZ)  # Sunday 9:00 AM


def at(hour, minute=0, days=0):
    return NOW.replace(hour=hour, minute=minute) + timedelta(days=days)


def e(title, hour, minute=0, days=0, end=None, place=None):
    return {"title": title, "start": at(hour, minute, days), "end": end, "place": place}


def test_a_time_is_said_the_way_a_person_says_it():
    assert [ear.ear_time(at(h, m)) for h, m in ((10, 40), (14, 0), (12, 0), (0, 0), (9, 0))] == ["10:40", "2", "noon", "midnight", "9"]
    # after noon a time with minutes keeps its PM: the voice reads a bare "2:30" as a 24-hour clock and would say "2:30 AM"
    assert ear.ear_time(at(14, 30)) == "2:30 PM" and ear.ear_time(at(23, 59)) == "11:59 PM"


def test_when_names_the_day_only_when_it_is_not_today_and_the_half_of_the_day_only_when_unclear():
    assert ear.ear_when(at(10, 40), NOW) == "at 10:40" and ear.ear_when(at(14), NOW) == "at 2"
    assert ear.ear_when(at(20), NOW) == "at 8 tonight" and ear.ear_when(at(20), NOW, tonight=False) == "at 8"
    assert ear.ear_when(at(3), NOW) == "at 3 AM" and ear.ear_when(at(2, 56), NOW) == "at 2:56 AM"
    assert ear.ear_when(at(14, days=1), NOW) == "tomorrow at 2" and ear.ear_when(at(12, days=2), NOW) == "Tuesday at noon"
    assert ear.ear_when(at(23, 59, days=2), NOW) == "Tuesday at 11:59 PM" and ear.ear_when(at(22, days=2), NOW) == "Tuesday night at 10"
    assert ear.ear_about(at(15, 2), NOW) == "around 3" and ear.ear_about(at(19, 41, days=-1), NOW) == "yesterday around 7:40 PM"
    assert ear.ear_about(at(19, 1, days=-1), NOW) == "yesterday evening around 7"


def test_a_title_loses_its_tags_and_a_course_code_is_said_by_name_when_there_is_one():
    assert ear.ear_title("COP4533 lecture") == "your COP 4533 lecture"
    assert ear.ear_title("COP4533 lecture", {"COP4533": "algorithms"}) == "your algorithms lecture"
    assert ear.ear_title("Knight Hacks: mentor check-in") == "the mentor check-in"
    assert ear.ear_title("Board game night (optional)") == "Board game night"
    assert ear.ear_title("Study session") == "your study session"
    assert ear.ear_title("Project - Milestone1 Fall2026 [COP4533]", article=False) == "Project Milestone 1"
    assert ear.ear_title("HW 4: linked lists", article=False) == "HW 4: linked lists"  # what is due keeps its whole name


def test_a_place_is_said_without_the_zone_or_camera_id():
    assert ear.ear_spot("desk", "cam-desk", "next to the coffee mug") == "on the desk, next to the coffee mug"
    assert ear.ear_spot(None, "cam-door") == "by the door" and ear.ear_spot(None, "cam0") == "in view of the camera"
    assert ear.said_last_seen("keys", at(15, 2), NOW, "desk", "cam-desk", "next to the mug") == "Your keys were last on the desk, next to the mug, around 3."


def test_a_rundown_is_sentences_three_things_at_most_then_an_offer_of_the_rest():
    day = [e("COP4533 lecture", 10, 40), e("PHY2049 lecture", 14), e("Knight Hacks: mentor check-in", 16, 30), e("Dinner with Maya", 19)]
    names = {"COP4533": "algorithms", "PHY2049": "physics"}
    assert ear.said_agenda(day[:2], NOW, names=names) == "You've got your algorithms lecture at 10:40, then your physics lecture at 2."
    assert ear.said_agenda(day[:3], NOW, names=names) == ("You've got your algorithms lecture at 10:40, your physics lecture at 2, "
                                                          "then the mentor check-in at 4:30 PM.")
    assert ear.said_agenda(day[:1], NOW) == "You've got your COP 4533 lecture at 10:40, and that's it for today."
    assert ear.said_agenda(day, NOW, names=names) == ("You've got four things today. First is your algorithms lecture at 10:40, "
                                                      "then your physics lecture at 2. Want the rest?")
    everything = ear.said_agenda(day, NOW, names=names, everything=True)
    assert everything.endswith("then Dinner with Maya at 7 tonight.") and "Want the rest" not in everything
    assert ear.said_agenda([], NOW) == "Nothing on your calendar today." and ear.said_agenda([], NOW, earlier=True) == "Nothing else on your calendar today."
    assert ear.said_agenda(day[1:2], NOW, "this afternoon", yes_no="have") == "Yes, you've got your PHY 2049 lecture at 2."
    assert ear.said_agenda([], NOW, "this afternoon", yes_no="have") == "No, nothing on your calendar this afternoon."
    assert ear.said_agenda(day[1:3], NOW, "this afternoon") == "This afternoon you've got your PHY 2049 lecture at 2, then the mentor check-in at 4:30 PM."


def test_things_a_connector_only_lists_are_on_campus_never_yours():
    campus = [e("U Can Finish Race", 7, 30, 1), e("Theatre UCF Open House", 11, 0, 1), e("Fall Climbing Competition: Dungeons & Dynos", 12, 0, 1),
              e("UCF Women's Soccer vs. Oklahoma State", 13, 0, 1), e("Graduate School Mixer", 15, 0, 1)]
    now = NOW + timedelta(days=1, hours=-8)  # that Monday, 1 AM
    said = ear.said_agenda(campus, now, listed=True)
    assert said == "There are five things on campus today. First is U Can Finish Race at 7:30, then Theatre UCF Open House at 11. Want the rest?"
    assert "you've got" not in said.lower() and ";" not in said and "Sun" not in said
    assert ear.said_agenda(campus[:2], now, listed=True) == "On campus today there's U Can Finish Race at 7:30, then Theatre UCF Open House at 11."


def test_next_when_free_and_due_answer_only_what_was_asked():
    lecture = e("COP4533 lecture", 10, 40, end=at(11, 30), place="L3Harris Engineering Center")
    assert ear.said_next(lecture, NOW) == "Next up is your COP 4533 lecture at 10:40."
    assert ear.said_next(lecture, NOW, where=True, ends=True) == "Next up is your COP 4533 lecture at 10:40. It's at L3Harris Engineering Center. It runs until 11:30."
    assert ear.said_next(None, NOW) == "Nothing else on your calendar today."
    assert ear.said_next(e("PHY2049 lecture", 14, days=1), NOW) == "Nothing else today. Next up is your PHY 2049 lecture tomorrow at 2."
    assert ear.said_when(lecture, NOW, "class") == "Your COP 4533 lecture is at 10:40." and ear.said_when(None, NOW, "exam") == "I don't see an exam on your calendar."
    assert ear.said_free(at(15), None, NOW) == "Yes, you're free at 3 this afternoon."
    assert ear.said_free(at(11), lecture, NOW) == "No, you've got your COP 4533 lecture then, until 11:30."
    assert ear.said_free(at(11), lecture, NOW, busy=True).startswith("Yes, you've got") and ear.said_free(at(15), None, NOW, busy=True).startswith("No, you're free")
    due = [e("COP 3502 HW 4: linked lists", 23, 59, 2), e("COP 3502 Exam 1", 10, 0, 4)]
    assert ear.said_due(due, NOW) == "You've got COP 3502 HW 4: linked lists due Tuesday at 11:59 PM, then COP 3502 Exam 1 Thursday at 10."
    assert ear.said_due([], NOW, "tomorrow") == "Nothing due tomorrow."


def test_hows_it_going_is_a_mood_its_work_for_you_and_an_offer():
    assert ear.said_done(["planned your day", "reminded you twice", "answered six questions"]) == (
        "Good. I've planned your day and reminded you twice so far. What else can I do for you?")
    assert ear.said_done([]) == "Good, all quiet so far. What can I do for you?"
    assert ear.said_done(["planned your day"], greeting=True) == "Hey. What can I do for you?"


def test_nothing_here_ever_reads_like_a_record():
    day = [e("COP4533 lecture [calendar]", 10, 40, end=at(11, 30), place="CSE A101"), e("PHY2049 lecture (Fall2026)", 14, end=at(15, 15))]
    for said in (ear.said_agenda(day, NOW), ear.said_next(day[0], NOW, where=True, ends=True), ear.said_free(at(11), day[0], NOW),
                 ear.said_visit(at(8, 55), NOW, ["A courier drops a box at the door."]), ear.said_due(day, NOW)):
        assert not re.search(r"\b[AP]M to\b|[\[\]()]|\bcam-|\b(?:Sun|Mon|Tue|Wed|Thu|Fri|Sat) \d|;", said), said
        assert said.endswith((".", "?"))
