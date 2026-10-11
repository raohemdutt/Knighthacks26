"""The check-in demo: an invented afternoon through the real app raises one check-in with every clause, and a
restart on that log shows it again without raising or saying anything. The look memory calls back to the earlier
afternoon the story holds, by the weekday it really was; `--onto` adds the same afternoon to the demo night."""

import asyncio
from datetime import datetime, timedelta

from contexts.demo import checkin
from contexts.demo.__main__ import main
from contexts.demo.seed import build_app, settle

END = datetime(2026, 10, 13, 15, 40, tzinfo=checkin.LOCAL)  # a Tuesday: three days back is a Saturday
THEN = END.date() - timedelta(days=checkin.THEN_BACK)
BACK = "That's the same look you had Saturday afternoon, when you'd gone two hours without moving."
HELPED = "Ten minutes away from the desk helped then."
# Aloud it is one look back: the callback, with the three earlier days left to their chip and the screen's paragraph.
# The advice is the rule's own, with what the look eased after last time said first.
SAID = ("You slept about five hours, your resting heart rate is up, and you've looked strained for the last twenty minutes. "
        f"{BACK} {HELPED} "
        "You've been at the desk for an hour and three quarters straight, and I haven't seen you pick up a drink since two o'clock. "
        "Take ten minutes outside, drink some water, and I can move the study block to after dinner if you say yes.")
ASKED = {"last": "The last time you looked this tired was Saturday afternoon, when you'd gone two hours without moving.",
         "helped": "Last time, Saturday afternoon, ten minutes away from the desk helped: the look eased within half an hour."}


def test_the_invented_afternoon_raises_one_check_in_with_every_clause_and_a_restart_only_shows_it(tmp_path):
    async def run():
        shown = await checkin.show(tmp_path, END)
        [e] = shown.checkins
        [told] = shown.told
        assert [s.kind for s in e.signals] == ["sleep", "resting_hr", "expression", "sitting", "drink", "history"]
        assert told.spoken == SAID and told.ref == e.event_id and e.proposes_replan and e.ts == END
        assert e.history.matches == 3 and [d.day for d in e.history.days] == [END.date() - timedelta(days=n) for n in (2, 4, 6)]
        assert "The last 3 times a day looked like this" in e.text and BACK in e.text  # the screen's paragraph keeps both
        # the look memory: the earlier afternoon is in the log as source events, and what is said of it is the memory's own
        back = e.callback
        assert (back.said, back.learned, back.lead, back.day) == (BACK, HELPED, "walk", THEN) and THEN.strftime("%A") == "Saturday"
        assert back.chip == "Looked tired · like Sat" and [s.kind for s in e.suggestions] == ["walk", "drink", "move_block"]
        assert back.lines == ["Saturday afternoon, around 2:45 PM. At the desk for two hours.",
                              "Ten minutes away from the desk helped: the look eased within half an hour."]
        looks = shown.app.day.looks
        then, now = looks.remembered(END)[1], looks.remembered(END)[0]
        assert (then.day, then.context.sitting_min, then.context.since_drink_min) == (THEN, 120, None) and now.checkin_id == e.event_id
        assert (then.outcome.got_up, then.outcome.away_min, then.outcome.eased, then.outcome.drink) == (True, 10, True, None)
        # that afternoon is the owner's looks and nothing else: no desk, drink or camera event that Habits counts from
        assert {x.type for x in checkin.earlier(END)} == {"perception.expression.observed"}
        assert {x.member_id for x in checkin.earlier(END)} == {checkin.MEMBER} and len({x.event_id for x in checkin.earlier(END)}) == len(checkin.earlier(END))
        assert {k: looks.said(k, END) for k in ASKED} == ASKED
        assert looks.said("why", END).endswith(BACK) and "looked tired for a while" in looks.said("why", END)
        assert "look memory: the earlier look was on Saturday 2026-10-10." in checkin.summary(shown)
        # the demo owner is on the People screen (made-up vectors, marked as demo data), scanned, with expressions on
        [owner] = [m for m in shown.app.identity.members() if m["member_id"] == checkin.MEMBER]
        assert owner["demo"] and owner["scanned"] and owner["expressions"]
        # nothing is nudged, closed or planned about that earlier day: Habits never heard of it
        written = list(shown.app.bus.read_log())
        assert not [x for x in written if x.type.startswith("habits.") and x.ts.astimezone(checkin.LOCAL).date() < END.date()]
        assert not [x for x in written if x.type.startswith("day.plan") and x.day != END.date()]
        assert not [x for x in written if x.type.startswith("day.look")]  # a projection: remembering writes no event
        assert SAID in checkin.summary(shown) and "\"Move the block\" offered: yes" in checkin.summary(shown)
        assert shown.app.day.planner.pending == {}  # nothing was proposed: the plan is as it was
        assert shown.app.memory.last_seen("bottle").camera_id == "cam-desk"
        log = [x.type for x in shown.app.bus.read_log()]
        assert log.count("day.checkin.raised") == 1

        # a restart: the same card from the log, nothing raised, said or pushed again
        again = build_app(tmp_path)
        seen = []
        again.bus.subscribe("day.checkin", seen.append)
        again.bus.subscribe("day.narrated", seen.append)
        await again.bus.replay_into()
        await again.bus.publish_all(again.day.tick(END + timedelta(seconds=30)))
        await settle()
        assert not [x for x in seen if x.type == "day.checkin.raised" or getattr(x, "moment", "") == "checkin"]
        card = again.day.checkin.card(again.day.checkin.latest(END + timedelta(minutes=1)))
        assert card["event_id"] == e.event_id and card["spoken"] == SAID
        assert set(card["evidence"]) == {"sleep", "resting_hr", "expression", "sitting", "drink", "history"}
        assert card["callback"]["chip"] == "Looked tired · like Sat" and card["callback"]["said"] == BACK
        assert again.day.looks.view(END)["moments"] == shown.app.day.looks.view(END)["moments"]  # the same memory from the log alone
        assert "values" not in card["evidence"]["resting_hr"] and len(card["evidence"]["history"]["days"]) == 3

    asyncio.run(run())


def test_at_night_or_in_the_lecture_nothing_is_raised_and_the_command_says_why(tmp_path, capsys):
    assert main(["checkin", "--data-dir", str(tmp_path / "night"), "--end", "2026-10-13T22:30:00-04:00"]) == 1
    out = capsys.readouterr().out
    assert "No check-in was raised." in out and "held: outside waking hours" in out
    assert main(["checkin", "--data-dir", str(tmp_path / "night"), "--end", "2026-10-13T15:40:00-04:00"]) == 1  # not an empty directory
    assert "already has an event log" in capsys.readouterr().err


def test_the_default_time_is_now_in_the_afternoon_and_twenty_to_four_otherwise():
    def local(h, m=0):
        return datetime(2026, 10, 13, h, m, tzinfo=checkin.LOCAL)

    assert checkin.default_end(local(14, 5)) == local(14, 5) and checkin.default_end(local(18, 59)) == local(18, 59)
    assert checkin.default_end(local(9)) == local(15, 40) and checkin.default_end(local(22)) == local(15, 40)


def test_the_weekday_is_the_one_it_really_was_whatever_day_the_story_ends(tmp_path):
    async def run():
        end = datetime(2026, 10, 16, 15, 40, tzinfo=checkin.LOCAL)  # a Friday: three days back is a Tuesday
        shown = await checkin.show(tmp_path, end)
        [e] = shown.checkins
        assert e.callback.said == "That's the same look you had Tuesday afternoon, when you'd gone two hours without moving."
        assert e.callback.day == datetime(2026, 10, 13).date() and e.callback.chip == "Looked tired · like Tue"

    asyncio.run(run())


def test_onto_the_demo_night_the_same_afternoon_calls_back_and_the_night_is_what_it_was(tmp_path):
    """`python -m contexts.demo circle` then `checkin --onto`: one box with the night, the Circle, the people and the
    check-in. The earlier afternoon lies before the night's first day, so Habits (whose days are closed) takes none of it."""
    from contexts.demo import circle
    from contexts.demo.seed import seed

    night = datetime(2026, 10, 13, 9, 30, tzinfo=checkin.LOCAL)

    async def run():
        cfg = circle.config_dir(tmp_path / "data" / "config")
        seeded = await seed(tmp_path / "data", now=night, config_dir=cfg, extra=circle.circle_events, history=True, people=True)
        def closed(app) -> dict:
            """The days the night had closed: each habit's value, whether it was met and its streak."""
            return {d: [(r["id"], r["value"], r["met"], r["streak"]) for r in app.habits.digest(day=d, now=night)["habits"]]
                    for d in (seeded.last_full_day, seeded.last_full_day - timedelta(days=1))}

        before = closed(seeded.app)
        standings = seeded.app.league.view(night)["standings"]
        assert seeded.written["day.checkin.raised"] == 0
        seeded.app.projection.store.close()

        try:  # too close to the night's last event: refused, with the reason, and nothing is written
            await checkin.show(tmp_path / "data", night + timedelta(minutes=30), onto=True)
            raise AssertionError("an afternoon that overlaps the night was not refused")
        except checkin.SeedRefused as refused:
            assert "the afternoon needs" in str(refused)

        # at 15:40 the night has a study session on (a fixed block: the rule stays quiet), so a clear time is picked
        held = [(m.title, m.start, m.end) for m in seeded.app.day.planner.current[night.date()].moves if m.fixed and m.start <= END < m.end]
        end = checkin.onto_end(tmp_path / "data", night)
        assert held and end.date() == END.date() and end != END and checkin.time(13, 30) <= end.time() < checkin.time(19, 0)
        assert not [m for m in seeded.app.day.planner.current[night.date()].moves if m.fixed and m.start - checkin.ONTO_GUARD <= end < m.end]
        shown = await checkin.show(tmp_path / "data", end, onto=True)
        [e] = shown.checkins
        [told] = shown.told
        assert e.ts == end and e.callback.said == BACK and e.callback.learned == HELPED and e.callback.day == THEN
        assert BACK in told.spoken and HELPED in told.spoken and told.spoken.count("same look") == 1
        app = shown.app
        assert app.circle is not None and {m["member_id"] for m in app.identity.members() if m.get("demo")} >= {checkin.MEMBER}
        # the days the night had already closed are exactly what they were, and so is last week's League
        assert closed(app) == before and all(before.values())
        assert [(r["display_name"], r["score"]) for r in app.league.view(night)["standings"]] == [(r["display_name"], r["score"]) for r in standings]
        days = [m["day"] for m in app.day.looks.view(end)["moments"]]
        assert days[0] == END.date().isoformat() and THEN.isoformat() in days and len(days) >= 3  # with the people's strained afternoons
        app.projection.store.close()

    asyncio.run(run())


def test_the_command_takes_onto_and_refuses_a_directory_with_no_night(tmp_path, capsys):
    assert main(["checkin", "--data-dir", str(tmp_path / "empty"), "--onto", "--end", "2026-10-13T15:40:00-04:00"]) == 1
    assert "has no event log to add the afternoon to" in capsys.readouterr().err
