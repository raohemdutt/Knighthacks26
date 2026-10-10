import asyncio
import copy
from bus import EventBus
from contexts.habits.domain.config import load_config
from contexts.habits.tests.helpers import (
    assignment, at, calendar, closed, day, disposed, feed, one_day, scene, signals, sit, tracker, zone,
)

IDS = ["sleep", "meals", "energy_drinks", "focus", "start_early", "leave_ready", "sort_right"]


def test_the_config_loads_seven_habits():
    cfg = load_config()
    assert [h.id for h in cfg.habits] == IDS
    assert {h.id: (h.target, h.direction) for h in cfg.habits} == {
        "sleep": (6, "at_least"), "meals": (2, "at_least"), "energy_drinks": (2, "at_most"), "focus": (2, "at_least"),
        "start_early": (24, "at_least"), "leave_ready": (0, "at_most"), "sort_right": (80, "at_least")}
    assert all(h.nudge and h.name for h in cfg.habits)
    assert cfg.by_id("meals").keywords and cfg.by_id("energy_drinks").keywords  # word lists live in the config
    assert cfg.by_id("leave_ready").watch_items == ["keys", "charger"]
    assert str(cfg.schedule.tz) == "America/New_York" and cfg.schedule.wake.hour == 8


# --- the seven habits ---


def test_sleep_is_the_longest_empty_desk_stretch_overnight():
    t = tracker()
    out = feed(t, sit(11, "21:00", "23:30") + sit(12, "08:30", "09:00"))  # away 23:30 to 08:30, window ends 08:00
    assert signals(out, "away_window") == [8.5]
    assert t.digest(day(12))["habits"][0]["value"] == 8.5
    out = feed(t, sit(12, "21:00", "22:00") + sit(13, "02:00", "03:00") + sit(13, "07:00", "07:30") + sit(14, "12:00", "12:30"))
    assert signals(out, "away_window")[:2] == [1.0, 4.0]  # 20:00 to 21:00, then 22:00 to 02:00
    assert closed(out, "sleep") == [(12, 8.5, True, 1), (13, 4.0, False, 0)]  # up at 2 and again at 7


def test_sleep_is_not_judged_on_a_night_atlas_did_not_watch():
    t = tracker()
    out = feed(t, sit(11, "14:00", "15:00") + sit(12, "14:00", "15:00"))
    assert closed(out, "sleep") == [(11, 0.0, True, 0)]  # first night unseen: met by default, no streak
    assert t.history["sleep"][0].applies is False


def test_meals_are_counted_once_per_sitting():
    t = tracker()
    out = feed(t, [
        scene(at(11, "12:00"), caption="A bowl of ramen next to the keyboard"),
        scene(at(11, "12:20"), caption="a half-eaten bowl of ramen"),  # same meal
        scene(at(11, "13:00"), caption="a bowling trophy on the shelf"),  # not a bowl
        scene(at(11, "19:00"), objects=["plate", "fork"], zone_id="desk"),
        scene(at(11, "19:05"), caption="a pizza box", camera="cam-door"),  # not the desk
    ])
    assert signals(out, "meal") == [1, 1]
    row = next(r for r in t.digest(day(11))["habits"] if r["id"] == "meals")
    assert (row["value"], row["met"]) == (2, True)


def test_energy_drinks_count_new_cans_only():
    t = tracker()
    out = feed(t, [
        scene(at(11, "10:00"), objects=["Red Bull can", "red bull can", "laptop"], zone_id="desk"),
        scene(at(11, "10:10"), objects=["red bull can", "Red Bull can"], zone_id="desk"),  # the same two
        scene(at(11, "10:20"), caption="three energy drinks on the desk", objects=["red bull", "celsius", "red bull"]),
        scene(at(11, "15:00"), caption="A soda can and a mug"),
    ])
    assert signals(out, "drink") == [2, 1]
    row = next(r for r in t.digest(day(11))["habits"] if r["id"] == "energy_drinks")
    assert (row["value"], row["target"], row["met"]) == (3, 2, False)


def test_focus_is_desk_time_on_a_day_with_a_deadline_near():
    t = tracker()
    out = feed(t, [
        assignment(at(11, "09:00"), "COP4533 Milestone 1", at(12, "23:59")),
        zone(at(11, "10:00"), "enter"),
        zone(at(11, "11:00"), "dwell", dwell_s=3600),
        zone(at(11, "10:30"), "enter", track=2),  # a friend joins: still one desk, not double time
        zone(at(11, "12:30"), "exit"),
        zone(at(11, "12:30"), "exit", track=2),
    ])
    assert signals(out, "due_soon") == [39.0]
    assert signals(out, "desk_time") == [150.0]
    row = next(r for r in t.digest(day(11))["habits"] if r["id"] == "focus")
    assert (row["value"], row["met"], row["applies"]) == (2.5, True, True)


def test_focus_is_not_judged_without_a_deadline():
    t = tracker()
    out = feed(t, sit(11, "10:00", "10:20") + sit(12, "10:00", "10:20"))
    assert closed(out, "focus") == [(11, 0.33, True, 0)]
    assert t.history["focus"][0].applies is False


def test_desk_time_across_midnight_lands_on_both_days():
    t = tracker()
    feed(t, sit(11, "23:00", "23:00") + [zone(at(11, "23:00"), "enter"), zone(at(12, "01:00"), "exit")])
    assert t.history["focus"][0].value == 1.0
    assert t.days[day(12)].desk_min == 60


def test_start_early_measures_the_lead_before_the_due_date():
    t = tracker()
    out = feed(t, [
        assignment(at(11, "09:00"), "Essay", at(13, "12:00"), "essay"),
        assignment(at(11, "09:00"), "Quiz", at(14, "12:00"), "quiz"),
        *sit(11, "10:00", "10:30"),  # counts for both: Atlas cannot tell which one is on the screen
        assignment(at(12, "09:00"), "Lab", at(14, "18:00"), "lab"),  # posted later, never touched
        assignment(at(13, "20:00"), "Pop quiz", at(14, "08:00"), "pop"),  # posted under 24 h ahead: not judged
        scene(at(15, "09:00")),
    ])
    assert signals(out, "start_lead") == [49.5, 73.5]
    assert closed(out, "start_early") == [(11, 0.0, True, 0), (12, 0.0, True, 0), (13, 49.5, True, 1), (14, 0.0, False, 0)]
    assert [c.applies for c in t.history["start_early"]] == [False, False, True, True]


def test_leave_ready_flags_a_watch_item_left_before_an_event():
    t = tracker()
    out = feed(t, [
        calendar(at(11, "07:00"), "Algorithms lecture", at(11, "09:00")),
        scene(at(11, "08:30"), objects=["car keys", "laptop charger", "mug"], zone_id="desk"),
        zone(at(11, "08:20"), "exit", zone_id="door"),  # 40 minutes out: too early to count
        zone(at(11, "08:40"), "exit", zone_id="door"),
        zone(at(11, "08:45"), "exit", zone_id="door"),  # back and out again: the same miss
    ])
    assert signals(out, "forgot") == [2]
    assert t.forgotten["items"] == ["charger", "keys"] and t.forgotten["title"] == "Algorithms lecture"
    row = next(r for r in t.digest(day(11))["habits"] if r["id"] == "leave_ready")
    assert (row["value"], row["met"]) == (2, False)


def test_leave_ready_is_met_when_the_items_went_along():
    t = tracker()
    out = feed(t, [
        calendar(at(11, "07:00"), "Algorithms lecture", at(11, "09:00")),
        scene(at(11, "08:30"), objects=["keys"], zone_id="desk"),
        scene(at(11, "08:38"), objects=["mug"], zone_id="desk"),  # keys picked up
        zone(at(11, "08:40"), "exit", zone_id="door"),
        scene(at(12, "09:00")),
    ])
    assert signals(out, "forgot") == []
    assert closed(out, "leave_ready") == [(11, 0, True, 1)]


def test_sort_right_is_the_share_of_correct_disposals():
    t = tracker()
    out = feed(t, [disposed(at(11, f"1{i}:00"), correct=i != 0, item_id=f"cam-door:{i}") for i in range(5)])
    assert signals(out, "disposal") == [0.0, 1.0, 1.0, 1.0, 1.0]
    row = next(r for r in t.digest(day(11))["habits"] if r["id"] == "sort_right")
    assert (row["value"], row["met"], row["applies"]) == (80, True, True)


# --- closing days ---


def test_days_close_on_event_time_with_streaks_and_a_missed_day():
    t = tracker()
    out = feed(t, one_day(11) + one_day(12, meals=1) + one_day(13))
    assert closed(out, "meals") == [(11, 2, True, 1), (12, 1, False, 0)]  # day 13 is still open
    out = feed(t, one_day(14) + one_day(15))
    assert closed(out, "meals") == [(13, 2, True, 1), (14, 2, True, 2)]
    assert closed(out, "energy_drinks") == [(13, 0, True, 3), (14, 0, True, 4)]
    assert len([e for e in out if e.type == "habits.day.closed"]) == 2 * len(IDS)  # one per habit per day


def test_a_gap_of_days_closes_every_day_in_between():
    t = tracker()
    out = feed(t, one_day(11) + one_day(14))
    assert closed(out, "meals") == [(11, 2, True, 1), (12, 0, False, 0), (13, 0, False, 0)]


def test_flush_closes_yesterday_after_the_grace_period():
    t = tracker()
    assert t.flush(at(11, "12:00")) == []  # nothing seen yet
    feed(t, one_day(11))
    assert t.flush(at(11, "23:59")) == [] and t.flush(at(12, "00:01")) == []
    out = t.flush(at(12, "00:03"))
    assert closed(out, "meals") == [(11, 2, True, 1)] and all(e.ts == at(12, "00:03") for e in out)
    assert t.flush(at(12, "00:04")) == []  # closed once
    assert feed(t, [scene(at(11, "23:58"), caption="a late pizza")]) == []  # too late: the day is closed


# --- replay ---


def script() -> list:
    return [
        assignment(at(11, "09:00"), "COP4533 Milestone 1", at(12, "23:59")),
        calendar(at(11, "07:00"), "Algorithms lecture", at(11, "13:00")),
        *sit(11, "10:00", "11:30"),
        scene(at(11, "12:00"), caption="a sandwich on a plate", objects=["keys", "monster energy can"], zone_id="desk"),
        zone(at(11, "12:40"), "exit", zone_id="door"),
        disposed(at(11, "12:41"), correct=True),
        *sit(11, "20:00", "23:00"),
        *sit(12, "08:10", "10:10"),
        *one_day(12),
        *one_day(13, meals=1),
        *sit(13, "09:00", "09:30"),
    ]


def state(t) -> tuple:
    return t.today, t.days, t.history, t.assignments, t.calendar, t.active_until, t.on_desk, t.digest(now=at(14, "09:00"))


def test_the_same_events_twice_change_nothing():
    t, events = tracker(), script()
    first = feed(t, events)
    before = state(copy.deepcopy(t))
    assert first and feed(t, events) == []  # no second signal, no second close
    assert state(t) == before
    fresh = tracker()
    again = feed(fresh, script_copy(events))
    assert state(fresh) == before
    assert [e.event_id for e in again] == [e.event_id for e in first]  # ids follow the source event


def script_copy(events) -> list:
    return [type(e).model_validate(e.model_dump()) for e in events]


def wire(bus: EventBus):
    t = tracker()
    bus.subscribe("perception.zone", t.on_zone_event, replay=True)
    bus.subscribe("perception.scene", t.on_scene, replay=True)
    bus.subscribe("connector.item.synced", t.on_item, replay=True)
    bus.subscribe("impact.item.disposed", t.on_disposed, replay=True)
    bus.subscribe("habits.day.closed", t.on_day_closed, replay=True)
    bus.subscribe("habits.nudge.raised", t.on_nudge, replay=True)
    return t


def test_state_rebuilds_from_the_log_without_duplicates(tmp_path):
    async def run():
        live_bus = EventBus(tmp_path)
        live = wire(live_bus)
        await live_bus.publish_all(script())
        await live_bus.publish_all(live.flush(at(14, "00:05")))  # the ticker closes day 13
        logged = [e.event_id for e in live_bus.read_log() if e.type.startswith("habits.")]
        assert len(logged) == len(set(logged)) and len(live.history["meals"]) == 3

        bus = EventBus(tmp_path)  # restart
        rebuilt = wire(bus)
        await bus.replay_into()
        assert state(rebuilt) == state(live)
        await bus.publish_all(rebuilt.flush(at(14, "00:06")))  # nothing left to close
        await bus.publish_all(script())  # a connector or camera re-sending old events
        assert [e.event_id for e in bus.read_log() if e.type.startswith("habits.")] == logged

    asyncio.run(run())


def test_late_events_for_a_closed_day_are_ignored():
    t = tracker()
    feed(t, one_day(11) + one_day(12))
    before = state(copy.deepcopy(t))
    assert feed(t, [scene(at(11, "22:00"), caption="a late-night burrito")]) == []
    assert state(t) == before


# --- digest ---


def test_digest_shape():
    t = tracker()
    feed(t, one_day(11) + one_day(12, meals=1))
    d = t.digest()
    assert d["day"] == "2026-10-12" and [r["id"] for r in d["habits"]] == IDS and d["latest_nudge"] is None
    meals = d["habits"][1]
    assert meals == {"id": "meals", "name": "Meals", "unit": "meals", "direction": "at_least", "target": 2,
                     "day": "2026-10-12", "value": 1, "met": False, "applies": True, "streak": 1, "closed": False,
                     "nudge": None}
    assert t.digest(day(11))["habits"][1]["closed"] is True
    assert t.digest(day(10))["habits"] == []  # before Atlas was watching
    assert [(r["day"], r["value"], r["closed"]) for r in t.history_for("meals", days=7)] == [
        ("2026-10-11", 2, True), ("2026-10-12", 1, False)]
    assert len(t.history_for("meals", days=1)) == 1


# --- a day the cameras did not watch (testing round 1, O1) ---


def test_a_connector_sync_does_not_make_a_day_or_a_night_watched():
    t = tracker()
    out = feed(t, [
        assignment(at(10, "10:30"), "COP4533 Milestone 1", at(11, "23:59")),  # the box syncs a day before any camera
        calendar(at(10, "10:30"), "Opening ceremony", at(11, "11:00")),
        zone(at(11, "12:02"), "enter", zone_id="door"),  # the first camera event
        *sit(11, "12:06", "12:30"),
        *one_day(12),
    ])
    # Friday had connector items only: nothing a camera measures is judged, and no streak is earned.
    for habit_id in ("sleep", "meals", "energy_drinks", "focus", "leave_ready"):
        day_10, day_11 = t.history[habit_id]
        assert (day_10.day, day_10.applies, day_10.met, day_10.streak) == (day(10), False, True, 0), habit_id
    # Saturday's night ended at 08:00, four hours before the first camera event: not 12 hours of sleep.
    assert signals(out, "away_window") == []
    saturday = t.history["sleep"][1]
    assert (saturday.value, saturday.applies, saturday.streak) == (0.0, False, 0)
    # The rest of Saturday was watched, so it is judged, and it is the first day of any streak.
    assert [(c.applies, c.streak) for c in t.history["leave_ready"]] == [(False, 0), (True, 1)]
    assert [(c.applies, c.met) for c in t.history["meals"]] == [(False, True), (True, False)]
    assert t.history["focus"][1].applies  # the deadline is near and the desk was in view


def test_start_early_is_not_judged_on_an_assignment_no_camera_could_have_seen_started():
    t = tracker()
    feed(t, [assignment(at(10, "09:00"), "Essay", at(12, "10:00")),  # known 49 hours ahead ...
             *sit(11, "20:00", "21:00"),  # ... but the cameras came on 14 hours before it was due
             *one_day(13)])
    assert [(c.day, c.applies) for c in t.history["start_early"]] == [(day(10), False), (day(11), False), (day(12), False)]
