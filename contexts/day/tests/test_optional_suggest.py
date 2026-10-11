"""Atlas brings an optional event up by itself (contexts/day/application/optional.py `tick`): once, between ninety and
thirty minutes before it, when it fits or nearly fits and names somebody, through the narrator and the arbiter. What
keeps it quiet, what a restart remembers, what the phone is pushed (never who is going), and the yes that follows."""

import asyncio
from datetime import datetime, timedelta

from fastapi.testclient import TestClient

from contexts.day.application.loop import day_phone, suggest_push
from contexts.day.domain import narration as words
from contexts.day.tests.test_day import zone
from contexts.day.tests.test_narration import StubRuntime
from contexts.day.tests.test_optional_events import (  # noqa: F401
    DINNER,
    DINNER_BLOCK,
    FRIENDS,
    MONDAY,
    REVIEW,
    SOCCER,
    Box,
    at,
    box,
    calendar,
    names_in,
    near,
    offer,
    places,
    said_yes,
)
from contracts.day import DayNarrated, DayOptionalAnswered, DayPlanCommitted, DayReplanProposed
from runtime.fake import FakeRuntime

SAID = ("Riley, Sam and Alex are having dinner at '63 South at quarter past five. It's a sixteen-minute walk and it fits before your study "
        "block. Want me to add it?")
SHOWN = "Friends' dinner: 5:15 PM to 6:15 PM at '63 South. 16 min walk, leave by 4:59 PM. Fits before your study block at 7:01 PM. Add it?"


class Ntfy:
    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []

    async def publish(self, topic, message, title=None, tags=None) -> None:
        self.sent.append((title, message))


def day_box(places, tmp_path, name: str = "a", items=None, day: dict | None = None, runtime=None) -> Box:  # noqa: F811
    """Monday with the morning plan made and a person at the desk since 3:40 PM; Atlas may suggest (the defaults)."""
    b = box(places, tmp_path, name=name, items=items, day=day or {}, runtime=runtime)
    b.morning()
    b.publish(zone(at(15, 40), "desk"))
    return b


def suggestions(b: Box) -> list[DayNarrated]:
    return [e for e in b.of(DayNarrated) if e.moment == "suggest"]


# ---------- when it is said ----------


def test_it_is_said_once_inside_the_window_and_not_before_or_after_it(places, tmp_path):  # noqa: F811
    b = day_box(places, tmp_path / "a")
    b.tick(at(15, 44))  # ninety-one minutes before: too early
    assert suggestions(b) == [] and b.row(DINNER, at(15, 44))["suggest_held"] == "not within the suggestion window"
    b.tick(at(16))
    [said] = suggestions(b)
    assert (said.ref, said.ts, said.day, said.plan_id) == (DINNER, at(16), MONDAY, b.plan().plan_id)
    assert said.event_id == f"day-narrated-suggest-{DINNER}" and said.spoken == SAID and said.display == SHOWN
    assert (said.engine, said.urgent, said.merged_into, said.spoken_by) == ("template", False, None, None)
    assert names_in(said.display) == [] and names_in(said.spoken) == ["Riley", "Sam", "Alex"]  # names are for the ear only
    assert words.check(said.spoken, b.optional.facts(b.optional.find(DINNER, at(16)), b.optional.fit(b.optional.find(DINNER, at(16)), at(16)), b.plan()),
                       said.spoken) == []
    row = b.row(DINNER, at(16))
    assert (row["suggested_at"], row["suggest_held"], row["state"]) == (at(16).isoformat(), "already suggested", "suggested")
    for minute in range(1, 45):  # not twice, however often the ticker looks
        b.tick(at(16, minute))
    assert suggestions(b) == [said]

    late = day_box(places, tmp_path / "late")
    late.tick(at(16, 46))  # twenty-nine minutes before: too late to bring up
    assert suggestions(late) == [] and late.row(DINNER, at(16, 46))["suggest_held"] == "not within the suggestion window"


def test_it_shows_in_the_brief_and_is_pushed_once_without_a_name(places, tmp_path):  # noqa: F811
    b = day_box(places, tmp_path)
    ntfy = Ntfy()
    b.bus.subscribe("day.", day_phone(ntfy, "atlas", narrated=True))
    b.tick(at(16))
    b.tick(at(16, 1))
    c = b.client()
    brief = c.get("/api/day/brief").json()
    assert brief["latest"]["suggest"]["ref"] == DINNER and brief["latest"]["suggest"]["spoken"] == SAID
    assert c.get("/api/day/brief?moment=suggest").json()["narration"]["display"] == SHOWN
    assert c.post("/api/day/brief", json={"moment": "suggest"}).status_code == 404  # said when it is due, never on demand
    near_you = [m for m in ntfy.sent if m[0] == "Near you"]
    assert near_you == [("Near you", "Atlas has a suggestion near you: Friends' dinner: 5:15 PM to 6:15 PM at '63 South. 16 min walk, "
                                     "leave by 4:59 PM. Fits before your study block at 7:01 PM. Open Atlas to add it.")]
    assert names_in(" ".join(text for _, text in ntfy.sent)) == []  # nothing that went through the notification service names a friend
    assert suggest_push(suggestions(b)[0]) == near_you[0][1]


def test_a_second_box_on_the_same_log_does_not_bring_it_up_again(places, tmp_path):  # noqa: F811
    b = day_box(places, tmp_path)
    b.tick(at(16))
    assert len(suggestions(b)) == 1
    again = box(places, tmp_path, name="b", day={})
    assert asyncio.run(again.bus.replay_into()) > 0 and again.out == []
    assert again.optional.suggested == b.optional.suggested == {DINNER: at(16)} and again.loop.narrator.latest["suggest"] == suggestions(b)[0]
    for minute in range(1, 40):
        again.tick(at(16, minute))
    assert suggestions(again) == [] and again.row(DINNER, at(16, 30))["suggest_held"] == "already suggested"
    log = [e for e in again.bus.read_log() if e.type == "day.narrated" and e.moment == "suggest"]
    assert len(log) == 1


def test_never_after_a_no_never_when_switched_off_and_never_with_the_narration_off(places, tmp_path):  # noqa: F811
    b = day_box(places, tmp_path / "no")
    events, _ = b.optional.answer(at(15, 50), DINNER, "no")
    b.publish(*events)
    off = day_box(places, tmp_path / "off", day={"suggest_optional": False})
    silent = day_box(places, tmp_path / "silent", day={"narration": "off"})
    for minute in range(0, 45, 5):
        for x in (b, off, silent):
            x.tick(at(16, minute))
    assert suggestions(b) == [] and b.row(DINNER, at(16, 20))["suggest_held"] == "declined"
    assert suggestions(off) == [] and off.row(DINNER, at(16, 20))["suggest_held"] == "suggestions are switched off"
    assert suggestions(silent) == [] and silent.row(DINNER, at(16, 20))["suggest_held"] == "narration is off"


def test_the_window_and_the_cooldown_come_from_the_config(places, tmp_path):  # noqa: F811
    b = day_box(places, tmp_path, day={"suggest_from_min": 60, "suggest_until_min": 20, "suggest_cooldown_min": 120})
    b.tick(at(16))
    assert suggestions(b) == []  # seventy-five minutes before: outside a sixty-minute window
    b.tick(at(16, 15))
    assert [e.ref for e in suggestions(b)] == [DINNER]
    for hour, minute in ((17, 30), (17, 45), (18, 5)):  # the review at 6:30 PM comes into its window inside the two-hour cooldown
        b.tick(at(hour, minute))
    assert [e.ref for e in suggestions(b)] == [DINNER]
    assert b.row(REVIEW, at(17, 45))["suggest_held"] == "something was suggested less than forty-five minutes ago"


# ---------- what keeps it quiet ----------


def test_one_suggestion_at_a_time_and_the_next_event_waits_out_the_cooldown(places, tmp_path):  # noqa: F811
    tea = offer("tea", at(17, 40), "6:10 PM", "Student Union")  # its window opens at 4:10 PM, the dinner's at 3:45 PM
    b = day_box(places, tmp_path, items=calendar() + near() + [tea])
    for minute in range(45, 60):
        b.tick(at(15, minute))
    for minute in range(0, 21):
        b.tick(at(16, minute))
    assert [e.ref for e in suggestions(b)] == [DINNER]
    assert b.row("tea", at(16, 20))["suggest_held"] == "something was suggested less than forty-five minutes ago"
    for minute in range(21, 60):
        b.tick(at(16, minute))
    assert [(e.ref, e.ts) for e in suggestions(b)] == [(DINNER, at(15, 45)), (tea.external_id, at(16, 30))]  # forty-five minutes apart


def test_nobody_named_a_fixed_block_running_and_a_time_to_leave_keep_it_quiet(places, tmp_path):  # noqa: F811
    lunch = offer("lunch-out", at(12, 30), "1:15 PM", "Student Union")  # its whole window (11:00 to 12:00) is the lecture and its end
    b = day_box(places, tmp_path, items=calendar() + near() + [lunch])
    b.publish(zone(at(10, 45), "desk", "exit"))
    for hour, minute in ((11, 0), (11, 30), (11, 54)):
        b.tick(at(hour, minute))
    assert suggestions(b) == [] and b.row("lunch-out", at(11, 30))["suggest_held"] == "during COP4533 lecture"
    assert b.row("resume-workshop", at(11, 30))["suggest_held"] in ("during COP4533 lecture", "nobody is named")
    b.tick(at(11, 56))  # the lecture is over: the workshop at 12:30 names nobody, the lunch has friends
    assert [e.ref for e in suggestions(b)] == [lunch.external_id]
    assert b.row("resume-workshop", at(11, 57))["suggest_held"] == "nobody is named"
    # an accepted event has a time to leave: nothing else is brought up in the half hour before it
    c = day_box(places, tmp_path / "c", items=calendar() + near() + [offer("tea", at(18, 30), "7:00 PM", "'63 South")])
    c.tick(at(15, 50))
    body = said_yes(c, DINNER, at(15, 52))
    c.approve(body["action_id"], at(15, 53))
    for hour, minute in ((17, 0), (17, 10)):  # the tea's window is open, and the dinner's leave-by was 4:59 PM: it has not started yet
        c.tick(at(hour, minute))
    assert [e.ref for e in suggestions(c)] == [DINNER] and c.row("tea", at(17, 5))["suggest_held"] == "close to a time to leave"


def test_an_open_card_holds_it_back_and_it_comes_once_the_card_is_gone(places, tmp_path):  # noqa: F811
    b = day_box(places, tmp_path)
    c = b.client()
    b.clock["now"] = at(15, 58)
    assert c.post("/api/day/plan", json={"from_here": True}).status_code == 200  # a re-plan the person asked for: its card is open
    b.tick(at(16))
    assert suggestions(b) == [] and b.row(DINNER, at(16))["suggest_held"] == "an approval card is open"
    b.tick(at(16, 2))
    assert suggestions(b) == []
    b.tick(at(16, 4))  # five minutes on the card is over; the dinner is still seventy-one minutes away
    assert [e.ref for e in suggestions(b)] == [DINNER] and suggestions(b)[0].ts == at(16, 4)


def test_due_with_a_block_start_it_is_said_as_the_end_of_that_line(places, tmp_path):  # noqa: F811
    tea = offer("tea", at(19, 40), "7:59 PM", "'63 South")  # its window opens at 6:10 PM; dinner in the kitchen starts at 6:30 PM
    b = box(places, tmp_path, items=calendar() + [tea], day={})
    ntfy = Ntfy()
    b.bus.subscribe("day.", day_phone(ntfy, "atlas", narrated=True))
    plan, card = b.loop.planner.plan(at(8, 10))
    b.publish(plan, card)
    b.approve(card.action_id, at(8, 11))  # reminders are for a plan that was approved
    assert b.of(DayPlanCommitted) and plan.plan_id in b.loop.planner.committed
    meal = next(m for m in b.plan().moves if m.kind == "eat" and m.start >= at(18))
    b.publish(zone(meal.start - timedelta(minutes=20), "desk"))
    before = len(b.out)
    b.tick(meal.start)  # one look: the meal starts and the tea comes into view
    told = [e for e in b.out[before:] if isinstance(e, DayNarrated) and e.moment in ("start", "suggest")]
    loud = [e for e in told if not e.merged_into]
    [inside] = [e for e in told if e.merged_into]
    assert [e.moment for e in loud] == ["start"] and loud[0].spoken.endswith("; and Tea with Maya is at '63 South at twenty to eight if you want it.")
    assert (inside.moment, inside.ref, inside.merged_into) == ("suggest", tea.external_id, loud[0].event_id)
    assert b.optional.suggested == {tea.external_id: meal.start}  # said, inside another line: it counts
    for minute in range(1, 30):
        b.tick(meal.start + timedelta(minutes=minute))
    assert len([e for e in b.of(DayNarrated) if e.moment == "suggest"]) == 1
    # the push is the block start's, once; the suggestion inside it is shown, and nothing pushed names Maya
    assert [title for title, _ in ntfy.sent if title in ("Next up", "Near you")] == ["Next up"]
    assert "Maya" not in " ".join(text for _, text in ntfy.sent) and "Tea: 7:40 PM to 7:59 PM" in loud[0].display


# ---------- the model's words ----------


def test_with_a_model_the_grounded_rewrite_is_said_once_and_an_invented_one_is_not(places, tmp_path):  # noqa: F811
    stub = StubRuntime()
    b = day_box(places, tmp_path / "a", runtime=stub)
    assert b.loop.narrator.engine == "model"
    before = len(stub.calls)
    for minute in range(0, 6):
        b.tick(at(16, minute))
    [said] = suggestions(b)
    assert said.engine == "model" and said.spoken == "Right, here's where things stand. " + SAID and said.display == SHOWN
    assert len(stub.calls) == before + 1 and b.optional.suggested == {DINNER: said.ts}

    def invented(draft: str) -> str:
        return draft.replace("Riley, Sam and Alex", "Riley, Sam, Alex and Jordan")

    c = day_box(places, tmp_path / "c", runtime=StubRuntime(invented))
    for minute in range(0, 6):
        c.tick(at(16, minute))
    [kept] = suggestions(c)
    assert (kept.engine, kept.spoken) == ("template", SAID)  # a name that is not in the facts: the template stands


# ---------- the yes that follows, and a yes given long before ----------


def test_suggested_then_yes_then_approved_and_nothing_more_is_said_about_it(places, tmp_path):  # noqa: F811
    b = day_box(places, tmp_path)
    c = b.client()
    b.tick(at(16))
    assert [e.ref for e in suggestions(b)] == [DINNER]
    b.clock["now"] = at(16, 0) + timedelta(seconds=20)  # "add it", twenty seconds after it was said
    yes = c.post("/api/day/optional/friends-dinner", json={"answer": "yes"}).json()
    assert yes["proposed"] and yes["action_id"] and b.plan().plan_id != yes["plan_id"]  # a card, not a change
    assert [a.answer for a in b.of(DayOptionalAnswered)] == ["yes"] and b.of(DayReplanProposed)[-1].for_optional == DINNER
    b.tick(at(16, 2))  # the global gap after the suggestion is over: the re-plan is said, in its own words
    told = [n for n in b.of(DayNarrated) if n.moment == "replan"][-1]
    assert told.spoken.startswith("Okay, I've worked Friends' dinner in: ") and told.ts >= at(16, 1, ) + timedelta(seconds=30)
    b.approve(yes["action_id"], at(16, 3))
    assert b.optional.accepted == {DINNER} and b.row(DINNER, at(16, 3))["state"] == "accepted"
    for minute in range(4, 45):
        b.tick(at(16, minute))
    assert [e.ref for e in suggestions(b)] == [DINNER] and b.row(DINNER, at(16, 30))["suggest_held"] == "accepted"


def test_a_yes_given_too_far_ahead_is_still_brought_up_before_it_starts(places, tmp_path):  # noqa: F811
    b = day_box(places, tmp_path)
    events, body = b.optional.answer(at(13), SOCCER, "yes")  # Tuesday 4:30 PM, asked Monday at 1 PM
    b.publish(*events)
    assert body["proposed"] is False and b.optional.answers[SOCCER].answer == "yes"
    tuesday = b.loop.planner.plan(datetime(2026, 10, 13, 8, 10, tzinfo=at(8).tzinfo))
    b.publish(*tuesday)
    b.publish(zone(at(14, 50, 13), "desk"))
    b.tick(at(15, 0, 13))
    b.tick(at(15, 1, 13))
    soccer = [e for e in suggestions(b) if e.ref == SOCCER]
    assert len(soccer) == 1 and soccer[0].spoken.startswith("Sam and Jordan are going to Pickup soccer at Recreation and Wellness Center at half past four.")


def test_without_a_runtime_or_with_the_fake_one_the_template_speaks(places, tmp_path):  # noqa: F811
    b = day_box(places, tmp_path, runtime=FakeRuntime())
    assert b.loop.narrator.engine == "template"
    b.tick(at(16))
    assert [(e.engine, e.spoken) for e in suggestions(b)] == [("template", SAID)]
