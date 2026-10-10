"""Nights and days the desk camera saw only part of: sleep and focus are judged on watched time only."""

import asyncio
from datetime import timedelta

from fastapi import FastAPI
from fastapi.testclient import TestClient

from bus import EventBus
from contexts.habits.api import habits_router
from contexts.habits.domain.coverage import Coverage
from contexts.habits.domain.members import MemberHabits
from contexts.habits.tests.helpers import (
    assignment, at, closed, day, feed, in_order, paused, signals, sit, tracker, watch, zone,
)
from contracts.events import Event


def row(t, habit_id, n=None) -> dict:
    return next(r for r in t.digest(day(n) if n else None)["habits"] if r["id"] == habit_id)


# --- the camera's coverage ---


def test_silence_is_a_gap_only_once_the_camera_has_reported_steadily():
    c = Coverage("cam-desk", gap=timedelta(minutes=5), steady=timedelta(minutes=30))
    for e in watch(at(11, "12:00"), at(11, "12:04")) + watch(at(11, "15:00"), at(11, "15:04")):  # two bursts: a script
        c.beat(e.ts)
    assert not c.measured and c.seen(at(11, "08:00"), at(11, "20:00"), at(11, "20:00")) == [(at(11, "08:00"), at(11, "20:00"))]
    for e in watch(at(11, "16:00"), at(11, "17:00")) + watch(at(11, "19:00"), at(11, "19:30")):  # a camera, off for two hours
        c.beat(e.ts)
    assert c.measured and c.steady_since == at(11, "16:00")
    assert c.seen(at(11, "14:00"), at(11, "19:30"), at(11, "19:30")) == [(at(11, "14:00"), at(11, "17:00")), (at(11, "19:00"), at(11, "19:30"))]
    # Four minutes after its last report it may still be on; after longer than the gap it has not been since then.
    assert c.seen(at(11, "19:00"), at(11, "19:34"), at(11, "19:34")) == [(at(11, "19:00"), at(11, "19:34"))]
    assert c.seen(at(11, "19:00"), at(11, "22:00"), at(11, "22:00")) == [(at(11, "19:00"), at(11, "19:30"))]
    assert not c.unbroken(at(11, "16:30"), at(11, "19:30")) and c.unbroken(at(11, "19:10"), at(11, "19:30"))


# --- sleep ---


def test_a_night_the_box_was_switched_on_in_is_not_judged():
    t = tracker()
    out = feed(t, in_order(watch(at(11, "02:00"), at(13, "00:10")),  # switched on at 2 AM
                           sit(11, "07:30", "07:40"), sit(11, "21:00", "23:00"), sit(12, "07:00", "07:30")))
    out += t.flush(at(13, "00:10"))
    assert closed(out, "sleep") == [(11, 5.5, True, 0), (12, 8.0, True, 1)]  # 5.5 empty hours seen: not a short night
    first, second = t.history["sleep"]
    assert (first.applies, first.coverage, first.reason) == (False, 0.5, "cameras saw 6 h of the 12 h night")
    assert (second.applies, second.coverage, second.reason) == (True, 1.0, None)


def test_an_empty_desk_does_not_run_across_a_stretch_no_camera_saw():
    events = [watch(at(10, "12:00"), at(11, "01:00")), watch(at(11, "03:00"), at(12, "00:10")),  # out from 1 to 3 AM
              sit(10, "20:30", "21:00"), sit(11, "09:00", "09:10")]
    t = tracker()
    out = feed(t, in_order(*events)) + t.flush(at(12, "00:10"))
    # 21:00 to 09:00 with two hours unseen is 4 hours, then 5: not the 11 an unbroken stretch would have been.
    assert signals(out, "away_window") == [5.0]
    assert closed(out, "sleep")[1] == (11, 5.0, False, 0)
    night = t.history["sleep"][1]
    assert (night.applies, night.coverage, night.reason) == (True, 0.83, None)


def test_a_night_mostly_unwatched_is_not_judged_and_keeps_the_streak():
    t = tracker()
    out = feed(t, in_order(watch(at(9, "18:00"), at(11, "01:00")), watch(at(11, "04:00"), at(12, "00:10")),  # out from 1 to 4 AM
                           sit(9, "18:30", "19:00"), sit(10, "09:00", "09:10"), sit(10, "20:30", "21:00"), sit(11, "09:00", "09:10")))
    out += t.flush(at(12, "00:10"))
    assert closed(out, "sleep")[1:] == [(10, 12.0, True, 1), (11, 4.0, True, 1)]  # neither a short night nor a full one
    night = t.history["sleep"][2]
    assert (night.applies, night.coverage, night.reason) == (False, 0.75, "cameras saw 9 h of the 12 h night")


def test_the_night_so_far_says_how_much_of_it_was_watched():
    t = tracker()
    feed(t, in_order(watch(at(10, "12:00"), at(10, "22:00")), sit(10, "19:00", "19:30")))
    t.flush(at(10, "22:03"))  # the camera is still reporting, as far as the gap allows
    assert [row(t, "sleep", 11)[k] for k in ("value", "applies", "coverage", "reason")] == [2.05, True, 1.0, None]
    t.flush(at(11, "02:00"))  # ... and has now been silent since ten
    assert [row(t, "sleep", 11)[k] for k in ("value", "applies", "coverage", "reason")] == [
        2.0, False, 0.33, "cameras saw 2 h of 6 h of the night so far"]


# --- focus ---


def test_focus_counts_desk_time_the_camera_saw_and_is_not_judged_on_a_day_it_mostly_missed():
    due = assignment(at(11, "07:00"), "COP4533 Milestone 1", at(12, "23:59"))
    desk = [zone(at(11, "10:00"), "enter"), zone(at(11, "13:40"), "exit")]
    t = tracker()
    out = feed(t, in_order(due, desk, watch(at(11, "07:00"), at(11, "10:30")), watch(at(11, "11:30"), at(12, "00:10"))))
    assert signals(out, "desk_time") == [160.0]  # 10:00 to 13:40 less the hour nobody saw
    assert [row(t, "focus", 11)[k] for k in ("value", "applies", "met", "coverage")] == [2.67, True, True, 0.82]  # of 08:00 to 13:40

    t = tracker()
    out = feed(t, in_order(due, desk, watch(at(11, "07:00"), at(11, "10:30")), watch(at(11, "13:30"), at(12, "00:10"))))
    out += t.flush(at(12, "00:10"))
    assert signals(out, "desk_time") == [40.0]
    assert closed(out, "focus") == [(11, 0.67, True, 0)]  # 40 minutes seen: not a missed day, and no streak for it
    focus = t.history["focus"][0]
    assert (focus.applies, focus.coverage, focus.reason) == (False, 0.75, "cameras saw 9 h of the 12 h day")


# --- cameras paused in the Vault policy ---


def paused_day() -> list:
    """Three days: the desk camera is paused from 7 PM on the 10th to 8:30 PM on the 11th."""
    return in_order(
        assignment(at(9, "17:00"), "COP4533 Milestone 1", at(12, "12:00")),
        watch(at(9, "18:00"), at(10, "19:00")), paused(at(10, "19:00"), "cam-desk"),
        paused(at(11, "20:30")), watch(at(11, "20:30"), at(13, "00:10")),
        sit(9, "18:30", "19:00"), sit(10, "09:00", "12:00"), sit(12, "09:00", "12:00"))


def test_a_day_with_the_desk_camera_paused_is_not_judged():
    t = tracker()
    out = feed(t, paused_day()) + t.flush(at(13, "00:10"))
    # The 11th: no desk time and no empty desk were seen. That is not a missed day and not a perfect night.
    assert closed(out, "sleep")[1:] == [(10, 12.0, True, 1), (11, 0.0, True, 1), (12, 11.5, True, 2)]
    assert closed(out, "focus")[1:] == [(10, 3.0, True, 1), (11, 0.0, True, 1), (12, 3.0, True, 2)]
    sleep, focus = t.history["sleep"][2], t.history["focus"][2]
    assert (sleep.applies, sleep.coverage, sleep.reason) == (False, 0.0, "cameras were paused for 12 h of the 12 h night")
    assert (focus.applies, focus.coverage, focus.reason) == (False, 0.0, "cameras were paused for 12 h of the 12 h day")
    assert [c.applies for c in t.history["meals"]] == [True, True, True, True]  # counts are judged as before


def test_a_paused_morning_reads_as_paused_not_as_a_sleepless_night():
    t = tracker()
    feed(t, [e for e in paused_day() if e.ts < at(11, "07:00")])
    t.flush(at(11, "07:00"))
    assert [row(t, "sleep")[k] for k in ("value", "met", "applies", "streak", "reason")] == [
        0.0, True, False, 1, "cameras were paused for 11 h of 11 h of the night so far"]


def test_a_pause_counts_in_a_log_too_sparse_to_measure_gaps():
    t = tracker()
    out = feed(t, in_order(sit(10, "18:00", "19:00"), paused(at(10, "19:30"), "cam-desk", "cam-door"),
                           paused(at(11, "07:00"), "cam-door"), paused(at(11, "07:00")), sit(11, "09:00", "09:10"),
                           sit(12, "09:00", "09:10")))
    assert closed(out, "sleep") == [(10, 0.0, True, 0), (11, 1.0, True, 0)]  # 07:00 to 08:00 is all that was seen
    assert t.history["sleep"][1].reason == "cameras were paused for 11 h of the 12 h night"


# --- replay, people, the phone ---


def wire(bus: EventBus):
    t = tracker()
    bus.subscribe("perception.observation", t.on_observation, replay=True)
    bus.subscribe("vault.policy.changed", t.on_policy, replay=True)
    bus.subscribe("perception.zone", t.on_zone_event, replay=True)
    bus.subscribe("connector.item.synced", t.on_item, replay=True)
    bus.subscribe("habits.day.closed", t.on_day_closed, replay=True)
    return t


def test_coverage_rebuilds_from_the_log(tmp_path):
    async def run():
        live_bus = EventBus(tmp_path)
        live = wire(live_bus)
        await live_bus.publish_all(paused_day())
        await live_bus.publish_all(live.flush(at(13, "00:10")))

        bus = EventBus(tmp_path)  # restart
        rebuilt = wire(bus)
        await bus.replay_into()
        assert rebuilt.history == live.history and rebuilt.history["sleep"][2].reason
        assert rebuilt.digest(now=at(13, "09:00")) == live.digest(now=at(13, "09:00"))

    asyncio.run(run())


def test_every_enrolled_person_is_judged_on_the_same_coverage():
    habits = MemberHabits(tracker())
    habits.on_identity(Event(type="perception.identity.changed", ts=at(9, "17:00"), enabled=True, members=["m-ana"]))
    assert habits.members["m-ana"].cover is habits.base.cover
    feed(habits, paused_day())
    habits.flush(at(13, "00:10"))
    assert habits.members["m-ana"].history["sleep"][2].reason == "cameras were paused for 12 h of the 12 h night"


def test_the_api_says_why_a_night_was_not_judged():
    t = tracker()
    feed(t, paused_day())
    t.flush(at(13, "00:10"))
    app = FastAPI()
    app.include_router(habits_router(t), prefix="/api")
    c = TestClient(app)
    history = {r["day"]: r for r in c.get("/api/habits/sleep/history").json()}
    assert (history["2026-10-11"]["coverage"], history["2026-10-11"]["reason"]) == (0.0, "cameras were paused for 12 h of the 12 h night")
    assert (history["2026-10-12"]["coverage"], history["2026-10-12"]["reason"]) == (0.96, None)
    today = {r["id"]: r for r in c.get("/api/habits/today").json()["habits"]}
    assert "coverage" in today["sleep"] and "coverage" not in today["meals"]  # durations only


def test_a_log_with_no_steady_camera_has_no_coverage_to_report():
    t = tracker()
    feed(t, in_order(watch(at(11, "21:00"), at(11, "21:04")), sit(11, "21:00", "21:03"),
                     watch(at(12, "08:30"), at(12, "08:34")), sit(12, "08:30", "08:33")))
    assert not t.cover.measured
    sleep = row(t, "sleep", 12)
    assert (sleep["value"], sleep["applies"]) == (10.95, True) and "coverage" not in sleep  # as before: silence is an idle room
