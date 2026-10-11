"""The check-in demo: an invented afternoon through the real app raises one check-in with every clause, and a
restart on that log shows it again without raising or saying anything."""

import asyncio
from datetime import datetime, timedelta

from contexts.demo import checkin
from contexts.demo.__main__ import main
from contexts.demo.seed import build_app, settle

END = datetime(2026, 10, 13, 15, 40, tzinfo=checkin.LOCAL)
SAID = ("You slept about five hours, your resting heart rate is up, and you've looked strained for the last twenty minutes. "
        "The last three times a day looked like this, your study blocks fell apart by four o'clock. "
        "You've been at the desk for an hour and three quarters straight, and I haven't seen you pick up a drink since two o'clock. "
        "Drink some water, take ten minutes outside, and I can move the study block to after dinner if you say yes.")


def test_the_invented_afternoon_raises_one_check_in_with_every_clause_and_a_restart_only_shows_it(tmp_path):
    async def run():
        shown = await checkin.show(tmp_path, END)
        [e] = shown.checkins
        [told] = shown.told
        assert [s.kind for s in e.signals] == ["sleep", "resting_hr", "expression", "sitting", "drink", "history"]
        assert told.spoken == SAID and told.ref == e.event_id and e.proposes_replan and e.ts == END
        assert e.history.matches == 3 and [d.day for d in e.history.days] == [END.date() - timedelta(days=n) for n in (2, 4, 6)]
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
