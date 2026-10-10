"""Sleep from a watch (K9): a `health` item's hours asleep replace the empty-desk estimate for that night."""

import asyncio
from datetime import timedelta

from fastapi import FastAPI
from fastapi.testclient import TestClient

from bus import EventBus
from contexts.habits.api import habits_router
from contexts.habits.application.nudges import Nudger
from contexts.habits.domain.members import MemberHabits
from contexts.habits.tests.helpers import (
    assignment, at, closed, day, feed, in_order, night, one_day, signals, sit, tracker, watch,
)
from contracts.connectors import ItemSynced
from contracts.events import Event


def row(t, habit_id="sleep", n=None, **kw) -> dict:
    return next(r for r in t.digest(day(n) if n else None, **kw)["habits"] if r["id"] == habit_id)


def camera_nights() -> list:
    """Three days at the desk: away 23:30 to 08:30, then up at 2 AM and 7 AM, then a long night."""
    return (sit(11, "21:00", "23:30") + sit(12, "08:30", "09:00") + sit(12, "21:00", "22:00") + sit(13, "02:00", "03:00")
            + sit(13, "07:00", "07:30") + sit(14, "12:00", "12:30"))


# --- with a watch night ---


def test_a_watch_night_is_the_sleep_value_instead_of_the_empty_desk():
    t = tracker()
    feed(t, sit(11, "21:00", "23:30") + sit(12, "08:30", "09:00"))  # desk empty 8.5 h of the window
    assert (row(t, n=12)["value"], row(t, n=12)["source"]) == (8.5, "camera")
    assert feed(t, [night(at(12, "09:05"), 12, "5:05")]) == []  # the watch: 5 h 5 min asleep. No signal, no desk event
    r = row(t, n=12)
    assert (r["value"], r["source"], r["met"], r["applies"], r["closed"]) == (5.08, "watch", False, True, False)
    out = feed(t, sit(13, "09:00", "09:10"))
    assert closed(out, "sleep") == [(12, 5.08, False, 0)]  # the day closes on what the watch measured
    done = t.history["sleep"][-1]
    assert (done.source, done.coverage, done.reason) == ("watch", None, None)
    assert row(t, n=12) | {"closed": True} == row(t, n=12) and row(t, n=12)["source"] == "watch"
    assert row(t, n=13)["source"] == "camera"  # the next night has no item: the desk again


def test_a_night_with_no_watch_item_behaves_exactly_as_before():
    plain, worn = tracker(), tracker()
    out_plain = feed(plain, camera_nights())
    # the watch was worn on the night ending the 13th only; the items for the 12th and 14th never come
    out_worn = feed(worn, in_order(camera_nights(), [night(at(13, "09:00"), 13, "6:30")]))
    assert closed(out_plain, "sleep") == [(11, 0.0, True, 0), (12, 8.5, True, 1), (13, 4.0, False, 0)]
    assert closed(out_worn, "sleep") == [(11, 0.0, True, 0), (12, 8.5, True, 1), (13, 6.5, True, 2)]
    assert signals(out_worn, "away_window") == signals(out_plain, "away_window")  # the desk's own signals are untouched
    for habit_id in plain.history:
        if habit_id != "sleep":
            assert worn.history[habit_id] == plain.history[habit_id]
    assert [(c.day, c.value, c.source) for c in worn.history["sleep"] if c.day != day(13)] \
        == [(c.day, c.value, c.source) for c in plain.history["sleep"] if c.day != day(13)]
    assert {c.source for c in plain.history["sleep"]} == {"camera"}
    # the night after is the desk's again, the same in both; only the streak it builds on differs
    assert row(worn, n=14) == row(plain, n=14) | {"streak": 2} and row(plain, n=14)["source"] == "camera"


def test_the_watch_night_is_judged_when_the_camera_rule_says_not_judged():
    events = in_order(watch(at(11, "02:00"), at(12, "00:10")), sit(11, "07:30", "07:40"))  # the box was switched on at 2 AM
    blind = tracker()
    out = feed(blind, events) + blind.flush(at(12, "00:10"))
    assert closed(out, "sleep") == [(11, 5.5, True, 0)]
    assert (blind.history["sleep"][0].applies, blind.history["sleep"][0].reason) == (False, "cameras saw 6 h of the 12 h night")

    worn = tracker()
    feed(worn, in_order(events, [night(at(11, "09:00"), 11, "4:40")]))
    r = row(worn, n=11, now=at(11, "12:00"))
    assert (r["value"], r["applies"], r["met"], r["source"]) == (4.67, True, False, "watch")
    assert "coverage" not in r and "reason" not in r  # the figure does not rest on the camera
    out = worn.flush(at(12, "00:10"))
    assert closed(out, "sleep") == [(11, 4.67, False, 0)]
    focus = next(r for r in worn.digest(day(11))["habits"] if r["id"] == "focus")
    assert "source" not in focus  # only the sleep row says where its number came from


def test_the_watch_night_is_judged_with_no_camera_at_all_and_starts_no_camera_clock():
    t = tracker()
    feed(t, [night(at(12, "09:00"), 12, "7:28")])
    assert t.watching_since is None
    rows = {r["id"]: r for r in t.digest(day(12), now=at(12, "12:00"))["habits"]}
    assert (rows["sleep"]["value"], rows["sleep"]["applies"], rows["sleep"]["met"], rows["sleep"]["source"]) == (7.47, True, True, "watch")
    assert not rows["meals"]["applies"] and not rows["energy_drinks"]["applies"]  # nothing was seen of the desk
    out = t.flush(at(13, "00:05"))
    assert closed(out, "sleep") == [(12, 7.47, True, 1)] and closed(out, "meals") == [(12, 0, True, 0)]


def test_a_later_upload_of_the_same_night_replaces_the_earlier_reading():
    t = tracker()
    feed(t, [night(at(12, "03:30"), 12, "3:10")])  # the phone uploaded in the middle of the night
    assert row(t, n=12, now=at(12, "03:30"))["value"] == 3.17
    feed(t, [night(at(12, "09:00"), 12, "7:02")])  # the whole night, same external id
    assert row(t, n=12, now=at(12, "09:00"))["value"] == 7.03
    again = night(at(12, "09:00"), 12, "7:02")
    assert feed(t, [again, again]) == [] and row(t, n=12, now=at(12, "09:30"))["value"] == 7.03


def test_a_week_of_nights_arriving_at_once_only_touches_days_still_open():
    t = tracker()
    out = feed(t, sit(11, "21:00", "23:30") + sit(12, "08:30", "09:00") + sit(13, "10:00", "10:30"))
    assert closed(out, "sleep") == [(11, 0.0, True, 0), (12, 8.5, True, 1)]
    before = list(t.history["sleep"])
    # the connector's first sync: every night of the last week, all stamped now
    feed(t, [night(at(13, "10:40"), d, asleep) for d, asleep in ((10, "6:00"), (11, "5:00"), (12, "4:00"), (13, "6:45"))])
    assert t.history["sleep"] == before  # the 11th and 12th are closed and in the log: they stay what they were
    assert t.measured_sleep == {day(13): 6.75}
    assert (row(t, n=13)["value"], row(t, n=13)["source"]) == (6.75, "watch")
    feed(t, sit(14, "09:00", "09:10"))
    assert t.measured_sleep == {}  # nothing is kept past the close


def test_an_item_that_is_not_a_night_is_ignored():
    t = tracker()
    feed(t, sit(11, "21:00", "23:30") + sit(12, "08:30", "09:00"))
    odd = [ItemSynced(ts=at(12, "09:05"), connector="health", kind="health", external_id="sleep:2026-10-12", title="Sleep", summary="no numbers"),
           ItemSynced(ts=at(12, "09:06"), connector="health", kind="health", external_id="workout:2026-10-12", title="Run",
                      summary="asleep 1 h 0 min"),
           ItemSynced(ts=at(12, "09:07"), connector="health", kind="health", external_id="sleep:2026-10-12", title="Sleep")]
    assert feed(t, odd) == [] and (row(t, n=12)["value"], row(t, n=12)["source"]) == (8.5, "camera")
    assert t.assignments == {} and t.calendar == {}


# --- replay ---


def story() -> list:
    return in_order(
        [assignment(at(11, "09:00"), "COP4533 Milestone 1", at(12, "23:59"))],
        camera_nights(), one_day(12), one_day(13, meals=1),
        [night(at(12, "09:10"), 12, "3:10"), night(at(12, "10:10"), 12, "7:02"),  # an early upload, then the whole night
         night(at(14, "09:00"), 14, "5:20"), night(at(14, "09:00"), 12, "9:00")],  # the 12th again, long after it closed
    )


def wire(bus: EventBus):
    t = tracker()
    bus.subscribe("perception.zone", t.on_zone_event, replay=True)
    bus.subscribe("perception.scene", t.on_scene, replay=True)
    bus.subscribe("connector.item.synced", t.on_item, replay=True)
    bus.subscribe("habits.day.closed", t.on_day_closed, replay=True)
    return t


def test_watch_nights_rebuild_identically_from_the_log(tmp_path):
    async def run():
        live_bus = EventBus(tmp_path)
        live = wire(live_bus)
        await live_bus.publish_all(story())
        await live_bus.publish_all(live.flush(at(15, "00:05")))
        logged = [e for e in live_bus.read_log() if e.type == "habits.day.closed" and e.habit_id == "sleep"]
        assert [(e.day.day, e.value, e.met, e.streak) for e in logged] == [
            (11, 0.0, True, 0), (12, 7.03, True, 1), (13, 4.0, False, 0), (14, 5.33, False, 0)]
        assert [c.source for c in live.history["sleep"]] == ["camera", "watch", "camera", "watch"]

        bus = EventBus(tmp_path)  # restart
        rebuilt = wire(bus)
        await bus.replay_into()
        assert rebuilt.history == live.history and rebuilt.measured_sleep == live.measured_sleep == {}
        assert rebuilt.digest(now=at(15, "09:00")) == live.digest(now=at(15, "09:00"))
        assert rebuilt.history_for("sleep") == live.history_for("sleep")
        await bus.publish_all(story())  # the connector re-sending after the restart: nothing new is logged
        closes = [e.event_id for e in bus.read_log() if e.type == "habits.day.closed"]
        assert len(closes) == len(set(closes)) and rebuilt.history == live.history

    asyncio.run(run())


def test_a_restart_in_the_middle_of_the_day_keeps_the_open_days_watch_night(tmp_path):
    async def run():
        live_bus = EventBus(tmp_path)
        live = wire(live_bus)
        await live_bus.publish_all(in_order(sit(11, "21:00", "23:30"), sit(12, "08:30", "09:00"), [night(at(12, "09:10"), 12, "6:12")]))
        bus = EventBus(tmp_path)
        rebuilt = wire(bus)
        await bus.replay_into()
        assert rebuilt.measured_sleep == {day(12): 6.2}
        assert row(rebuilt, n=12) == row(live, n=12) and row(rebuilt, n=12)["source"] == "watch"

    asyncio.run(run())


# --- the nudge, the API, and whose watch it is ---


def test_a_short_measured_night_is_mentioned_in_the_watchs_words_not_the_desks():
    t = tracker()
    n = Nudger(t.cfg, t)
    feed(t, sit(11, "21:00", "23:00") + sit(12, "07:00", "07:10") + [night(at(12, "07:30"), 12, "4:30")])
    assert n.check(at(12, "07:45")) == []  # before wake time the ticker is quiet
    [nudge] = n.check(at(12, "08:05"))
    assert nudge.message == "Your watch measured 4.5 hours of sleep last night. Aim for 6 tonight."
    assert "bpm" not in nudge.message and nudge.evidence == []

    rested = tracker()
    feed(rested, sit(11, "21:00", "23:00") + sit(12, "03:00", "04:00") + [night(at(12, "07:30"), 12, "7:30")])
    assert Nudger(rested.cfg, rested).check(at(12, "08:05")) == []  # the desk says 4 h, the watch says 7.5: no nudge


def test_the_api_says_where_the_sleep_figure_came_from():
    t = tracker()
    app = FastAPI()
    app.include_router(habits_router(t), prefix="/api")
    c = TestClient(app)
    feed(t, sit(11, "21:00", "23:30") + sit(12, "08:30", "09:00") + [night(at(12, "09:05"), 12, "7:28")] + sit(13, "09:00", "09:10"))
    today = {r["id"]: r for r in c.get("/api/habits/today").json()["habits"]}
    assert today["sleep"]["source"] == "camera" and all("source" not in r for k, r in today.items() if k != "sleep")
    history = c.get("/api/habits/sleep/history").json()
    assert [(r["day"], r["value"], r["source"], r["closed"]) for r in history] == [
        ("2026-10-11", 0.0, "camera", True), ("2026-10-12", 7.47, "watch", True), ("2026-10-13", 12.0, "camera", False)]
    body = c.get("/api/habits/today").text + c.get("/api/habits/sleep/history").text
    assert "bpm" not in body and "HRV" not in body  # heart numbers stay in the item; Habits keeps the hours only


def identity(ts, enabled, members):
    return Event(type="perception.identity.changed", ts=ts, enabled=enabled, members=list(members))


def test_one_enrolled_person_gets_the_watch_night_and_several_do_not():
    solo = MemberHabits(tracker())
    solo.on_identity(identity(at(12, "08:00"), True, ["m-ana"]))
    solo.on_item(night(at(12, "09:00"), 12, "6:40"))
    assert (row(solo, n=12, member="m-ana")["value"], row(solo, n=12, member="m-ana")["source"]) == (6.67, "watch")
    assert row(solo.base, n=12)["source"] == "watch"  # the unnamed set keeps it too: it is what closes into the log

    two = MemberHabits(tracker())
    two.on_identity(identity(at(12, "08:00"), True, ["m-ana", "m-ben"]))
    two.on_item(night(at(12, "09:00"), 12, "6:40"))
    assert [row(two, n=12, member=m)["source"] for m in ("m-ana", "m-ben")] == ["camera", "camera"]  # whose watch? nobody's
    assert (row(two.base, n=12)["value"], row(two.base, n=12)["source"]) == (6.67, "watch")

    late = MemberHabits(tracker())
    late.on_item(night(at(12, "09:00"), 12, "6:40"))
    late.on_identity(identity(at(12, "10:00"), True, ["m-ana"]))  # enrolled after the night was synced
    assert row(late, n=12, member="m-ana")["source"] == "watch"

    off = MemberHabits(tracker())  # recognition off: the plain tracker
    off.on_item(night(at(12, "09:00"), 12, "6:40"))
    assert (row(off, n=12)["value"], row(off, n=12)["source"]) == (6.67, "watch") and off.members == {}
