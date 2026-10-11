"""Block-start reminders (contexts/day/domain/reminders.py, application/reminders.py) and the arbiter end to end:
said once when a planned block starts, through the arbiter; the busy quarter hour."""

import asyncio
from datetime import date, datetime, timedelta

from fastapi.testclient import TestClient

from contexts.day.domain import narration as words
from contexts.day.domain import reminders as rule
from contexts.day.tests.test_day import CLOCK, TZ, items, places  # noqa: F401 (fixtures)
from contexts.day.tests.test_narration import FakeNtfy, api, build, settle
from contexts.day.application.loop import day_phone
from contracts.day import (CheckinSignal, CheckinSuggestion, DayCheckinRaised, DayNarrated, DayPlanCommitted, DayPlanProposed,
                           DayStatusChanged, NextEvent, PlanMove, PlanOutcome)
from runtime.fake import FakeRuntime

MONDAY = date(2026, 10, 12)
DESK, UNION, LIBRARY, DINING = "apartment.desk", "ucf.union", "ucf.library", "ucf.dining_63"
NAMES = {DESK: "Desk", UNION: "Student Union", LIBRARY: "John C. Hitt Library", DINING: "'63 South"}


def at(hour: int, minute: int = 0, second: int = 0) -> datetime:
    return datetime(2026, 10, 12, hour, minute, second, tzinfo=TZ)


def move(i: str, kind: str, start: datetime, minutes: int, place: str | None = DESK, fixed: bool = False, title: str | None = None,
         origin: str | None = None) -> PlanMove:
    return PlanMove(id=i, kind=kind, title=title or kind.title(), start=start, end=start + timedelta(minutes=minutes), minutes=minutes,
                    place=place, place_name=NAMES.get(place), fixed=fixed, origin=origin)


# A Monday: two short study blocks with a break between them, lunch at the Union, a long library block, a class.
DAY = [move("m0", "study", at(14), 25), move("m1", "rest", at(14, 25), 15), move("m2", "study", at(14, 40), 25),
       move("m3", "eat", at(15, 30), 30, place=UNION, title="Lunch"), move("m4", "study", at(16, 30), 90, place=LIBRARY),
       move("class-1", "class", at(18, 15), 60, place=LIBRARY, fixed=True, title="COP4533 lecture")]


def walk(a, b) -> int:
    return 0 if a == b else 12


def due(now, moves=DAY, seen=DESK, committed=True, said=None, awake=True):
    return rule.due(moves, "p1", committed, now, seen, walk, said or rule.Said(set(), set()), TZ, awake=awake)


# --- the rule ---


def test_short_blocks_back_to_back_are_one_line_and_all_count_as_said():
    d = due(at(14))
    assert (d.first.id, d.count, d.until, d.there, d.walk) == ("m0", 2, at(15, 5), True, 0)
    assert d.ref == "p1:m0+m1+m2@14:00" and rule.parse(d.ref) == ("p1", ["m0", "m1", "m2"], "14:00")
    said = rule.Said(set(), set())
    said.note(d.ref, at(14), MONDAY)
    assert due(at(14, 25), said=said) is None and due(at(14, 40), said=said) is None  # the break and the second block: said


def test_a_block_that_needs_a_walk_is_said_at_the_time_to_leave():
    said = rule.Said(set(), set())
    said.note("p1:m0+m1+m2@14:00", at(14), MONDAY)
    assert due(at(15, 17), said=said) is None
    d = due(at(15, 18), said=said)  # 15:30 minus a twelve-minute walk
    assert (d.first.id, d.walk, d.there) == ("m3", 12, False)
    assert due(at(15, 23), said=said) is None  # the five-minute window closed


def test_continuing_the_same_thing_at_the_same_place_is_not_announced():
    moves = [move("a", "study", at(10), 60), move("b", "study", at(11), 60)]
    assert due(at(10), moves=moves).first.id == "a"
    said = rule.Said(set(), set())
    said.note("p1:a@10:00", at(10), MONDAY)
    assert due(at(11), moves=moves, said=said) is None  # still at the desk, carrying on
    assert due(at(11), moves=moves, said=said, seen=None).first.id == "b"  # not seen there: it is a new start


def test_ten_minutes_between_two_reminders_quiet_hours_and_never_during_a_fixed_event():
    moves = [move("a", "rest", at(12), 5), move("b", "eat", at(12, 5), 30)]
    said = rule.Said(set(), set())
    said.note("p1:a@12:00", at(12), MONDAY)
    assert due(at(12, 5), moves=moves, said=said) is None  # five minutes after the last one
    assert due(at(20, 30), moves=[move("late", "study", at(20, 30), 60)], awake=False) is None
    during = [move("lec", "class", at(9), 90, fixed=True), move("s", "study", at(10), 30)]
    assert due(at(10), moves=during) is None


def test_only_a_committed_plans_chosen_blocks_and_accepted_optional_events_close_by():
    assert due(at(14), committed=False) is None
    near = [move("optional:quiz-night", "event", at(19), 60, place=DESK, fixed=True, origin="optional", title="Quiz night")]
    assert due(at(19), moves=near, committed=False).first.id == "optional:quiz-night"
    far = [move("optional:dinner", "event", at(17, 15), 60, place=DINING, fixed=True, origin="optional", title="Friends' dinner")]
    assert due(at(17, 3), moves=far) is None and due(at(17, 15), moves=far) is None  # the leave-now line is its reminder


def test_a_replan_landing_mid_window_does_not_announce_the_same_start_again():
    said = rule.Said(set(), set())
    said.note("p1:m0+m1+m2@14:00", at(14), MONDAY)
    replanned = [m.model_copy(update={"id": "n" + m.id[1:]}) for m in DAY]
    assert rule.due(replanned, "p2", True, at(14, 2), DESK, walk, said, TZ) is None


# --- through the arbiter, on the bus ---


def plan(moves=DAY, plan_id="p1") -> DayPlanProposed:
    out = PlanOutcome(won=True, stars=3, points=80, summary="", sleep_hours=8, meals=3, study_minutes=140, energy_min=0.5, deadline_risk=0)
    return DayPlanProposed(event_id=f"plan-{plan_id}", ts=at(8), day=MONDAY, plan_id=plan_id, start=at(8), end=at(20), moves=moves,
                           outcome=out, text="The plan.")


async def committed(loop, bus, moves=DAY) -> None:
    await bus.publish(plan(moves))
    await bus.publish(DayPlanCommitted(event_id="commit-p1", ts=at(8, 1), day=MONDAY, plan_id="p1", action_id="card-p1"))
    await settle()


async def tick(loop, bus, now) -> list:
    loop.tracker.seen = (DESK, "desk camera", now)
    out = loop.tick(now)
    await bus.publish_all(out)
    await settle()
    return [e for e in out if isinstance(e, DayNarrated)]


def test_a_block_start_is_said_once_pushed_once_and_shown_in_the_brief(items, places, tmp_path):  # noqa: F811
    loop, bus, seen = build(items, places, tmp_path, runtime=FakeRuntime())
    ntfy = FakeNtfy()
    bus.subscribe("day.", day_phone(ntfy, "atlas", narrated=True))

    async def scenario():
        await committed(loop, bus)
        first = await tick(loop, bus, at(14))
        again = [await tick(loop, bus, at(14, m)) for m in (1, 2, 3)]
        return first, again

    first, again = asyncio.run(scenario())
    starts = [e for e in first if e.moment == "start"]
    assert len(starts) == 1 and not any(e.moment == "start" for batch in again for e in batch)
    said = starts[0]
    assert said.spoken.startswith("It's two o'clock: two short study blocks") and said.spoken.endswith("You're at the desk already.")
    assert said.ref == "p1:m0+m1+m2@14:00" and said.event_id == "day-narrated-start-p1:m0+m1+m2@14:00" and said.ts == at(14)
    assert [m for m in ntfy.sent if m[0] == "Next up"] == [("Next up", said.display)]
    brief = TestClient(api(loop, bus)).get("/api/day/brief").json()
    assert brief["latest"]["start"]["ref"] == said.ref
    assert TestClient(api(loop, bus)).get("/api/day/brief?moment=start").json()["narration"]["spoken"] == said.spoken


def test_a_second_box_on_the_same_log_says_nothing_again(items, places, tmp_path):  # noqa: F811
    loop, bus, _ = build(items, places, tmp_path, runtime=FakeRuntime())

    async def first():
        await committed(loop, bus)
        return await tick(loop, bus, at(14))

    assert [e.moment for e in asyncio.run(first()) if e.moment == "start"] == ["start"]
    again, bus2, seen = build(items, places, tmp_path, runtime=FakeRuntime())

    async def restarted():
        await bus2.replay_into()
        return [await tick(again, bus2, at(14, m)) for m in (1, 2, 4)]

    assert not any(e.moment == "start" for batch in asyncio.run(restarted()) for e in batch)
    assert again.reminders.said.blocks == {"p1:m0", "p1:m1", "p1:m2"} and again.narrator.latest["start"].ref == "p1:m0+m1+m2@14:00"


def test_block_reminders_off_and_narration_off_say_nothing(items, places, tmp_path):  # noqa: F811
    for i, narration in enumerate(({"block_reminders": False}, {"narration": "off"})):
        loop, bus, _ = build(items, places, tmp_path / str(i), runtime=FakeRuntime(), narration=narration)

        async def scenario():
            await committed(loop, bus)
            return await tick(loop, bus, at(14))

        assert not [e for e in asyncio.run(scenario()) if e.moment == "start"]


def test_a_block_start_whose_lead_time_falls_inside_a_fixed_event_is_said_when_that_event_ends(items, places, tmp_path):  # noqa: F811
    """A lecture until 4:50 PM, then a study block at the library at 5:00 PM. The cameras last saw the person at the desk,
    before the lecture, and from the desk the lead time for the walk is inside the lecture. The reminder is not lost and
    is never said during the lecture: it comes when the lecture ends, with the walk from where the plan has the person."""
    before = at(15, 30)  # the cameras' last sighting: at the desk, before the lecture
    lecture = move("lec", "class", at(16), 50, place=UNION, fixed=True, title="COP4533 lecture")
    moves = [lecture, move("s", "study", at(17), 60, place=LIBRARY)]

    def late(now, moves=moves, seen_at=before):
        return rule.due(moves, "p1", True, now, DESK, walk, rule.Said(set(), set()), TZ, seen_at=seen_at)

    assert [late(at(16, m)) for m in (40, 48, 49)] == [None, None, None]  # 4:48 PM is the lead time: the lecture is running
    d = late(at(16, 50))  # "you're out at 4:50": said as the lecture ends, ten minutes before the block
    assert (d.first.id, d.walk, d.there, d.ref) == ("s", 12, False, "p1:s@17:00")
    assert late(at(16, 54)).first.id == "s" and late(at(16, 55)) is None  # for five minutes from then, and once
    # a fixed event that runs right up to the block: at the block's start, not a minute before
    tight = [lecture.model_copy(update={"end": at(17), "minutes": 60}), moves[1]]
    assert late(at(16, 59), tight) is None and late(at(17), tight).first.id == "s"
    # two fixed events back to back: after the second one
    office = move("oh", "class", at(16, 50), 5, place=UNION, fixed=True, title="Office hours")
    assert late(at(16, 50), [lecture, office, moves[1]]) is None and late(at(16, 55), [lecture, office, moves[1]]).first.id == "s"
    # seen at the desk during the lecture (the person never went): still not during it, and the walk is from the desk
    home = late(at(16, 50), seen_at=at(16, 20))
    assert (home.first.id, home.walk) == ("s", 12) and late(at(16, 49), seen_at=at(16, 20)) is None

    # through the loop: one line, at the lecture's end, in the narrator's own "start" words (no wording of its own)
    loop, bus, _ = build(items, places, tmp_path, runtime=FakeRuntime())
    real = [move("lec", "class", at(16), 50, place="ucf.cb1", fixed=True, title="COP4533 lecture"),
            move("s", "study", at(16, 55), 60, place=LIBRARY)]  # thirteen minutes from the desk, five from the classroom

    async def afternoon():
        await committed(loop, bus, real)
        told = []
        for minute in range(40, 58):
            loop.tracker.seen = (DESK, "desk camera", before)
            out = loop.tick(at(16, minute))
            await bus.publish_all(out)
            await settle()
            told += [e for e in out if isinstance(e, DayNarrated) and e.moment == "start"]
        return told

    [said] = asyncio.run(afternoon())
    assert (said.ts, said.ref) == (at(16, 50), "p1:s@16:55")
    facts = {"moment": "start", "block": "your study block", "kind": "study", "at": "16:55", "min": 60, "place": "John C. Hitt Library", "walk_min": 5}
    assert said.spoken == words.render(facts, loop.narrator.words).spoken and said.display == words.render(facts, loop.narrator.words).display


# --- the busy quarter hour ---

DINNER = NextEvent(id="dinner-1", title="Friends' dinner", start=at(17, 15), end=at(18, 15), place=DINING, place_name="'63 South",
                   walk_minutes=12, leave_by=at(17, 3))
SUGGEST = {"moment": "suggest", "title": "Board game night", "who": ["Riley"], "at": "18:30", "place": "the desk", "fits": "yes"}
LATER = {"moment": "suggest", "title": "Food truck night", "who": ["Sam"], "at": "18:45", "place": "Memory Mall", "fits": "yes"}


def checkin(ts: datetime) -> DayCheckinRaised:
    return DayCheckinRaised(event_id="checkin-1", ts=ts, day=MONDAY, at=ts, plan_id="p1",
                            signals=[CheckinSignal(kind="sitting", summary="At the desk 100 min", minutes=100)],
                            suggestions=[CheckinSuggestion(kind="walk", text="Take ten minutes outside")])


def leave_now(ts: datetime) -> DayStatusChanged:
    return DayStatusChanged(event_id="status-leave", ts=ts, day=MONDAY, plan_id="p1", now=ts, seen_at=DESK, seen_at_name="Desk",
                            next_event=DINNER, risk="leave_now")


def test_the_busy_quarter_hour(items, places, tmp_path):  # noqa: F811
    """A block start, a suggestion, a check-in and a leave-now all due between 4:30 and 4:45 PM: what is said, in
    what order, and what is dropped."""
    moves = [move("s1", "study", at(15, 30), 60), move("s2", "study", at(16, 30), 60, place=DESK, title="Study at the Library")]
    moves[1] = moves[1].model_copy(update={"kind": "eat", "title": "Snack", "minutes": 20, "end": at(16, 50)})
    loop, bus, seen = build(items, places, tmp_path, runtime=FakeRuntime())

    async def quarter_hour():
        await committed(loop, bus, moves)
        before = len(seen)
        # 4:30:00, one tick: the snack starts and Atlas has a suggestion (an optional-event producer calls tell in the tick)
        loop.narrator.tell("suggest", SUGGEST, at(16, 30), MONDAY, "p1", "nearby:board-game-night")
        await tick(loop, bus, at(16, 30))
        # 4:30:45: a check-in is raised: inside the global gap, so it waits
        await bus.publish(checkin(at(16, 30, 45)))
        await settle()
        await tick(loop, bus, at(16, 31))
        # 4:31:10: time to leave for dinner: urgent, it breaks the gap
        await bus.publish(leave_now(at(16, 31, 10)))
        await settle()
        await tick(loop, bus, at(16, 32))  # the slot is the urgent line's until 4:32:40: the check-in still waits
        await tick(loop, bus, at(16, 33))  # the check-in goes out
        # 4:34: another suggestion comes due: within fifteen minutes of a check-in it is dropped
        loop.narrator.tell("suggest", LATER, at(16, 34), MONDAY, "p1", "nearby:food-truck-night")
        await tick(loop, bus, at(16, 34))
        for m in range(35, 46):
            await tick(loop, bus, at(16, m))
        return seen[before:]

    said = asyncio.run(quarter_hour())
    loud = [e for e in said if not e.merged_into and (e.moment != "now" or e.urgent)]
    assert [(e.moment, e.ts) for e in loud] == [("start", at(16, 30)), ("now", at(16, 31, 10)), ("checkin", at(16, 33))]
    first = loud[0]
    assert first.spoken == "It's half past four: an afternoon meal, twenty minutes. You're at the desk already; and Board game night " \
        "with Riley is at the desk at half past six if you want it."  # the start's words, with the suggestion as its last clause
    [merged] = [e for e in said if e.merged_into]
    assert (merged.moment, merged.ref, merged.merged_into) == ("suggest", "nearby:board-game-night", first.event_id)
    assert loud[1].urgent and "dinner" in loud[1].display.lower()
    assert not [e for e in said if e.ref == "nearby:food-truck-night"]  # dropped: a check-in four minutes ago
    assert ("day-narrated-suggest-nearby:food-truck-night", "drop", "a check-in card is open") in loop.arbiter.log  # raised, not answered
    non_urgent = [e.ts for e in loud if not e.urgent]
    assert all((b - a).total_seconds() >= loop.arbiter.rules.gap_s for a, b in zip(non_urgent, non_urgent[1:]))


def test_one_speaker_the_box_says_the_line_and_it_is_marked_for_the_phone_to_stay_silent(items, places, tmp_path):  # noqa: F811
    from contexts.day.application.loop import box_voice

    loop, bus, _ = build(items, places, tmp_path, runtime=FakeRuntime())
    loop.arbiter.speaker = lambda: "box"
    heard: list[str] = []

    async def say(text: str) -> None:
        heard.append(text)

    bus.subscribe("day.narrated", box_voice(say))

    async def scenario():
        await committed(loop, bus)
        return await tick(loop, bus, at(14))

    [start] = [e for e in asyncio.run(scenario()) if e.moment == "start"]
    assert start.spoken_by == "box" and heard[-1] == start.spoken
    assert len([t for t in heard if t == start.spoken]) == 1
