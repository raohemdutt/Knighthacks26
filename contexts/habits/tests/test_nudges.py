from fastapi import FastAPI
from fastapi.testclient import TestClient

from contexts.habits.api import habits_router
from contexts.habits.application.nudges import Nudger
from contexts.habits.tests.helpers import assignment, at, calendar, disposed, feed, one_day, scene, sit, tracker, zone


def pair():
    t = tracker()
    return t, Nudger(t.cfg, t)


def live(t, n, events) -> list:
    """Feed events and let the nudger see what the tracker published, as the bus does live."""
    out = []
    for event in events:
        for e in feed(t, [event]):
            raised = n.on_signal(e) if e.type == "habits.signal.recorded" else n.on_day_closed(e)
            for nudge in raised:
                t.on_nudge(nudge)
            out += raised
    return out


def cans(clock: str, count: int):
    return scene(at(11, clock), objects=["red bull can"] * count, zone_id="desk")


def test_a_third_energy_drink_warns_once_per_cooldown_in_event_time():
    t, n = pair()
    assert live(t, n, [cans("10:00", 2)]) == []  # at the limit, not over it
    [warn] = live(t, n, [cans("10:10", 3)])
    assert (warn.habit_id, warn.tone, warn.ts) == ("energy_drinks", "warn", at(11, "10:10"))
    assert warn.message == "That is energy drink number 3 today. Your limit is 2."
    assert live(t, n, [cans("10:20", 4)]) == []  # 10 event-minutes later: still cooling down
    assert live(t, n, [cans("13:09", 1)]) == []  # a fresh can, 179 minutes after the warning
    [again] = live(t, n, [cans("13:11", 2)])
    assert again.message.startswith("That is energy drink number 6")


def test_the_cooldown_survives_a_restart():
    t, n = pair()
    live(t, n, [cans("10:00", 3)])
    fresh = Nudger(t.cfg, t)  # only the tracker's replayed nudge log remembers
    assert fresh.on_signal(feed(t, [cans("10:30", 4)])[0]) == []


def test_focus_nudge_joins_the_camera_with_the_assignment():
    t, n = pair()
    feed(t, [assignment(at(11, "09:00"), "COP4533 Milestone 1", at(12, "14:00")), *sit(11, "10:00", "10:20")])
    [nudge] = n.check(at(11, "12:00"))
    assert nudge.habit_id == "focus" and nudge.tone == "warn" and nudge.ts == at(11, "12:00")
    assert nudge.message == "COP4533 Milestone 1 is due in 26 hours and you have 20 focused minutes today"
    canvas, camera = nudge.evidence
    assert canvas.url == "https://canvas.test/a1" and camera.frame_ref.startswith("frames:cam-desk:")
    assert n.check(at(11, "13:00")) == []  # cooldown
    feed(t, sit(11, "13:00", "15:00"))
    assert [x.habit_id for x in n.check(at(11, "15:30"))] == ["meals"]  # 2 h 20 at the desk: focus is met


def test_the_ticker_raises_one_nudge_at_a_time_and_none_at_night():
    t, n = pair()
    feed(t, [assignment(at(11, "09:00"), "Essay", at(13, "09:30"))])
    assert n.check(at(11, "09:59")) == []  # focus and start-early wait until 10:00
    assert [x.habit_id for x in n.check(at(11, "10:00"))] == ["focus"]
    assert n.check(at(11, "10:10")) == []  # 20 minutes between ticker nudges
    [early] = n.check(at(11, "10:20"))
    assert early.habit_id == "start_early" and early.message.startswith("Essay is due in 47 hours and Atlas has not seen you")
    [meals] = n.check(at(11, "15:00"))
    assert meals.message == "Atlas has seen 0 meals at your desk today. Time to eat something."
    assert n.check(at(11, "22:00")) == [] and n.check(at(12, "03:00")) == []  # asleep


def test_a_short_night_is_mentioned_after_waking():
    t, n = pair()
    feed(t, sit(11, "21:00", "23:00") + sit(12, "03:00", "04:00"))
    assert n.check(at(12, "07:00")) == []
    [nudge] = n.check(at(12, "08:05"))
    assert nudge.message == "Your desk was empty for only 4 hours last night. Aim for 6 tonight."


def test_leaving_without_the_keys_names_the_calendar_event():
    t, n = pair()
    [nudge] = live(t, n, [
        calendar(at(11, "07:00"), "Algorithms lecture", at(11, "09:00")),
        scene(at(11, "08:30"), objects=["keys", "charger"], zone_id="desk"),
        zone(at(11, "08:40"), "exit", zone_id="door"),
    ])
    assert nudge.message == "You left your charger and keys on the desk and Algorithms lecture starts in 20 minutes"
    assert nudge.evidence[0].frame_ref.startswith("frames:cam-desk:") and nudge.evidence[1].title == "Algorithms lecture"


def test_a_wrong_bin_warns_only_while_the_day_is_below_target():
    t, n = pair()
    assert live(t, n, [disposed(at(11, "10:00"), True, "i1")]) == []
    [nudge] = live(t, n, [disposed(at(11, "10:05"), False, "i2")])
    assert nudge.message.startswith("Only 50% of what you threw away today")


def test_a_three_day_streak_is_celebrated_once():
    t, n = pair()
    out = live(t, n, [e for d in range(11, 17) for e in one_day(d)])  # closes days 11 to 15
    cheers = [(x.habit_id, x.ts.day) for x in out if x.tone == "celebrate" and x.habit_id == "meals"]
    assert cheers == [("meals", 14)]  # day 13 closed, streak 3; not again at 4 or 5
    assert next(x.message for x in out if x.habit_id == "meals") == "Meals: 3 days in a row. Keep it going."
    assert not any(x.habit_id in ("focus", "sort_right") for x in out)  # nothing judged, nothing cheered


def test_a_missed_day_restarts_the_count_to_three():
    t, n = pair()
    days = one_day(11) + one_day(12) + one_day(13, meals=1) + one_day(14) + one_day(15) + one_day(16) + one_day(17)
    cheers = [x.ts.day for x in live(t, n, days) if x.habit_id == "meals"]
    assert cheers == [17]  # 14, 15, 16 met


def test_the_api_serves_today_and_history():
    t, n = pair()
    live(t, n, one_day(11) + [cans("20:00", 3)] + one_day(12, meals=1))
    app = FastAPI()
    app.include_router(habits_router(t), prefix="/api")
    client = TestClient(app)
    today = client.get("/api/habits/today").json()
    assert today["day"] == "2026-10-12" and len(today["habits"]) == 7 and today["latest_nudge"] is None
    assert {"id", "name", "value", "target", "met", "streak", "nudge"} <= set(today["habits"][0])
    yesterday = t.digest(at(11).date())
    assert yesterday["latest_nudge"]["tone"] == "warn" and yesterday["habits"][2]["nudge"]["message"].startswith("That is")
    history = client.get("/api/habits/energy_drinks/history", params={"days": 7}).json()
    assert [(r["day"], r["value"], r["met"], r["streak"]) for r in history] == [("2026-10-11", 3, False, 0), ("2026-10-12", 0, True, 0)]
    assert client.get("/api/habits/nope/history").status_code == 404


def test_on_the_bus_nudges_are_raised_live_and_never_on_replay(tmp_path):
    """The wiring bootstrap uses: projection handlers replay, nudge decisions do not."""
    import asyncio

    from bus import EventBus
    from contexts.habits.tests.test_tracker import wire

    def wired(bus):
        t = wire(bus)
        n = Nudger(t.cfg, t)
        bus.subscribe("habits.signal.recorded", n.on_signal)
        bus.subscribe("habits.day.closed", n.on_day_closed)
        return t, n

    async def run():
        bus = EventBus(tmp_path)
        t, n = wired(bus)
        await bus.publish_all([cans("10:00", 3), assignment(at(11, "10:30"), "COP4533 Milestone 1", at(12, "14:00"))])
        await bus.publish_all(t.flush(at(11, "12:00")) + n.check(at(11, "12:00")))  # one tick
        raised = [e for e in bus.read_log() if e.type == "habits.nudge.raised"]
        assert [e.habit_id for e in raised] == ["energy_drinks", "focus"]
        assert t.digest()["latest_nudge"]["message"].startswith("COP4533 Milestone 1 is due in 26 hours")

        again = EventBus(tmp_path)  # restart
        t2, n2 = wired(again)
        await again.replay_into()
        assert len([e for e in again.read_log() if e.type == "habits.nudge.raised"]) == 2
        assert t2.digest() == t.digest()
        assert [x.habit_id for x in n2.check(at(11, "12:30"))] == ["start_early"]  # focus is still cooling down

    asyncio.run(run())
