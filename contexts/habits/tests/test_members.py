"""Habits per enrolled person (P9). Recognition events are built as plain events, the way the log gives them back."""

import asyncio
from datetime import timedelta

from fastapi import FastAPI
from fastapi.testclient import TestClient

from bus import EventBus
from contexts.habits.api import habits_router
from contexts.habits.application.nudges import Nudger
from contexts.habits.domain.config import load_config
from contexts.habits.domain.members import MemberHabits
from contexts.habits.tests.helpers import assignment, at, calendar, day, disposed, feed, one_day, scene, sit, tracker, zone
from contracts.events import Event

IDS = ["sleep", "meals", "energy_drinks", "focus", "start_early", "leave_ready", "sort_right"]
ANA, BEN = "m-ana", "m-ben"


def identity(ts, enabled, members):
    return Event(type="perception.identity.changed", ts=ts, enabled=enabled, members=list(members))


def tagged(ts, track, member, camera="cam-desk"):
    return Event(type="perception.person.identified", ts=ts, camera_id=camera, track_id=track, member_id=member)


def lost(ts, track=None, camera="cam-desk"):
    return Event(type="perception.person.lost", ts=ts, camera_id=camera, track_id=track)


def book(nudges=False) -> MemberHabits:
    cfg = load_config()
    return MemberHabits(tracker(cfg), nudger_for=(lambda t: Nudger(cfg, t)) if nudges else None)


def route(h, events) -> list:
    """Route events the way the bus does; returns everything that would be published."""
    handlers = {"perception.zone.event": h.on_zone_event, "perception.scene.described": h.on_scene,
                "connector.item.synced": h.on_item, "impact.item.disposed": h.on_disposed,
                "perception.identity.changed": h.on_identity, "perception.person.identified": h.on_person,
                "perception.person.lost": h.on_person}
    out = []
    for e in events:
        out += handlers[e.type](e) or []
    return out


def row(h, habit_id, d, member=None, now=None):
    return next(r for r in h.digest(day(d), now, member=member)["habits"] if r["id"] == habit_id)


def stay(d, start, end, track, member=None, camera="cam-desk"):
    """A person at the desk from `start` to `end`; if `member`, the track is tagged a few seconds after it appears."""
    a, b = at(d, start), at(d, end)
    tag = [tagged(a + timedelta(seconds=5), track, member, camera)] if member else []
    return [zone(a, "enter", track=track), *tag,
            zone(a + timedelta(seconds=9), "dwell", track=track, dwell_s=9), zone(b, "exit", track=track)]


def a_full_story():
    """Three days of everything the tracker counts, with nobody enrolled."""
    return (
        [assignment(at(11, "09:00"), "COP4533 Milestone 1", at(12, "23:59")), calendar(at(11, "09:05"), "COP 3502 lecture", at(11, "14:00"))]
        + sit(11, "10:00", "12:30") + [scene(at(11, "12:00"), caption="A bowl of ramen next to the keyboard")]
        + [scene(at(11, "13:00"), objects=["keys", "Red Bull can", "red bull can"], zone_id="desk")]
        + [zone(at(11, "13:40"), "exit", "door"), disposed(at(11, "15:00"), True), disposed(at(11, "15:05"), False)]
        + sit(11, "21:00", "23:30") + sit(12, "08:30", "11:00") + one_day(12) + sit(12, "23:50", "23:59")
        + [zone(at(13, "00:10"), "dwell", dwell_s=1200, track=4), zone(at(13, "00:20"), "exit", track=4)] + one_day(13, meals=1)
        + sit(14, "09:00", "09:30")
    )


# --- off: exactly as before ---


def test_with_recognition_off_everything_is_exactly_what_the_plain_tracker_gives():
    plain, off, nobody = tracker(), book(nudges=True), book(nudges=True)
    story = a_full_story()
    expected = [e.model_dump_json() for e in feed(plain, story)]
    assert expected and [e.model_dump_json() for e in route(off, story)] == expected
    # Switched on with nobody enrolled is still one unnamed set of habits: there is no one to attribute to.
    assert [e.model_dump_json() for e in route(nobody, [identity(at(11, "08:00"), True, [])] + story)] == expected
    for h in (off, nobody):
        assert not h.attributing and h.members == {}
        for d in (11, 12, 13, 14):
            assert h.digest(day(d), at(14, "10:00")) == plain.digest(day(d), at(14, "10:00"))
        assert [h.history_for(i) for i in IDS] == [plain.history_for(i) for i in IDS]
        closing = [e.model_dump_json() for e in h.flush(at(15, "00:05"))]
        assert closing == [e.model_dump_json() for e in tracker_copy(plain).flush(at(15, "00:05"))]


def tracker_copy(t):
    import copy

    return copy.deepcopy(t)


# --- on: only tagged tracks count, each member on their own ---


def test_desk_time_and_meals_count_only_for_the_enrolled_person_whose_track_it_is():
    h = book()
    out = route(h, [
        identity(at(11, "08:00"), True, [ANA, BEN]),
        assignment(at(11, "08:30"), "COP4533 Milestone 1", at(12, "23:59")),
        *stay(11, "10:00", "12:00", track=1, member=ANA),
        zone(at(11, "10:30"), "enter", track=2), tagged(at(11, "10:30"), 2, BEN),
        scene(at(11, "10:45"), caption="two plates of pasta on the desk"),  # both are at the desk
        zone(at(11, "11:00"), "dwell", track=2, dwell_s=1800), zone(at(11, "11:30"), "exit", track=2),
        scene(at(11, "11:45"), objects=["Red Bull can"], zone_id="desk"),  # only Ana is
        *stay(11, "13:00", "17:00", track=3),  # a guest: never tagged
        scene(at(11, "14:00"), caption="a pizza box and an energy drink", objects=["monster can", "pizza"]),
        scene(at(11, "19:00"), caption="a sandwich on the desk"),  # nobody is there
    ])
    # Nothing is published about a person: only the deadline, which came from Canvas.
    assert [e.kind for e in out if e.type == "habits.signal.recorded"] == ["due_soon"]
    now = at(11, "20:00")
    assert [(row(h, i, 11, ANA, now)["value"]) for i in ("meals", "energy_drinks", "focus")] == [1, 1, 2.0]
    assert [(row(h, i, 11, BEN, now)["value"]) for i in ("meals", "energy_drinks", "focus")] == [1, 0, 1.0]
    assert all(row(h, i, 11, m, now)["applies"] for i in ("meals", "energy_drinks", "focus") for m in (ANA, BEN))
    # The unnamed set saw none of it, and is not judged on a day when people are told apart.
    desk = h.base.digest(day(11), now)["habits"]
    assert [(r["value"], r["applies"]) for r in desk if r["id"] in ("meals", "energy_drinks", "focus")] == [(0, False)] * 3
    assert h.digest(day(11), now) == h.base.digest(day(11), now)  # two enrolled: no member asked for, the unnamed set


def test_time_before_the_face_was_matched_counts_from_when_that_track_sat_down():
    h = book()
    route(h, [identity(at(11, "08:00"), True, [ANA]), zone(at(11, "10:00"), "enter", track=1),
              zone(at(11, "10:30"), "dwell", track=1, dwell_s=1800),  # half an hour with her back to the camera
              scene(at(11, "10:40"), caption="a bowl of noodles"),  # arrives before the match: nobody's
              tagged(at(11, "10:45"), 1, ANA),
              zone(at(11, "11:00"), "dwell", track=1, dwell_s=3600), scene(at(11, "11:10"), caption="a plate of pasta"),
              lost(at(11, "11:30"), 1),  # the tag ends (the tracker lost her) before the zone engine says exit
              zone(at(11, "12:00"), "exit", track=1), scene(at(11, "12:10"), caption="a sandwich")])
    assert h.digest(day(11), at(11, "13:00"))["habits"][2]["id"] == "energy_drinks"  # one enrolled: hers by default
    assert h.view() is h.members[ANA]
    assert h.members[ANA].days[day(11)].desk_min == 60 and row(h, "meals", 11, ANA, at(11, "13:00"))["value"] == 1


def test_leaving_without_the_keys_is_the_tagged_persons_and_never_a_guests():
    h = book(nudges=True)
    start = [identity(at(11, "08:00"), True, [ANA]), calendar(at(11, "09:00"), "COP 3502 lecture", at(11, "14:00")),
             *stay(11, "10:00", "13:30", track=1, member=ANA),
             scene(at(11, "13:20"), objects=["keys", "laptop"], zone_id="desk")]
    route(h, start)
    guest = route(h, [zone(at(11, "13:40"), "exit", "door", track=5)])
    assert guest == [] and h.members[ANA].days[day(11)].forgot == 0
    out = route(h, [tagged(at(11, "13:44"), 6, ANA, camera="cam-door"), zone(at(11, "13:45"), "exit", "door", track=6)])
    assert [e.type for e in out] == ["habits.nudge.raised"] and "keys" in out[0].message and "lecture" in out[0].message
    assert row(h, "leave_ready", 11, ANA, at(11, "14:00"))["value"] == 1
    assert "Ana" not in out[0].model_dump_json() and ANA not in out[0].model_dump_json()  # a nudge names nobody


def test_the_third_can_nudges_for_the_enrolled_person_and_a_guests_cans_do_not():
    h = book(nudges=True)
    cans = lambda t, n: scene(at(11, t), objects=["red bull can"] * n, zone_id="desk")  # noqa: E731
    out = route(h, [identity(at(11, "08:00"), True, [ANA]), *stay(11, "09:00", "10:00", track=9), cans("09:30", 5)])
    assert out == []  # five cans in front of a guest
    out = route(h, [*stay(11, "11:00", "16:00", track=1, member=ANA)[:-1], cans("11:30", 2), cans("15:00", 1)])
    assert [(e.type, e.habit_id) for e in out] == [("habits.nudge.raised", "energy_drinks")] and "3" in out[0].message


def test_each_member_has_their_own_days_and_streaks():
    h = book()
    events = [identity(at(10, "20:00"), True, [ANA, BEN])]
    for d in (11, 12, 13):
        events += stay(d, "08:30", "20:30", track=10 * d + 1, member=ANA)
        events += [scene(at(d, "12:00"), caption="a plate of pasta"), scene(at(d, "19:00"), caption="a bowl of rice")]
        events += stay(d, "21:00", "22:00", track=10 * d + 2, member=BEN)  # Ben is only there after dinner
    events += stay(14, "09:00", "09:30", track=141, member=ANA)
    closes = [e for e in route(h, events) if e.type == "habits.day.closed"]
    assert {e.day.day for e in closes} == {11, 12, 13} and len(closes) == 3 * len(IDS)  # the unnamed set's, one per day
    assert all(e.met and e.streak == 0 for e in closes if e.habit_id == "meals")  # ... which are not judged

    assert [(c.day.day, c.value, c.met, c.streak) for c in h.members[ANA].history["meals"]] == [
        (11, 2, True, 1), (12, 2, True, 2), (13, 2, True, 3)]
    assert [(c.day.day, c.value, c.met, c.streak) for c in h.members[BEN].history["meals"]] == [
        (11, 0, False, 0), (12, 0, False, 0), (13, 0, False, 0)]
    # Sleep (the window is 20:00 to 08:00): Ana is away from 20:30, Ben from 22:00.
    assert [(c.day.day, c.value) for c in h.members[ANA].history["sleep"]][1:] == [(12, 11.5), (13, 11.5)]
    assert [(c.day.day, c.value) for c in h.members[BEN].history["sleep"]][1:] == [(12, 10.0), (13, 10.0)]
    assert [r["streak"] for r in h.history_for("meals", 7, member=ANA)] == [1, 2, 3, 3]
    assert [r["streak"] for r in h.history_for("meals", 7, member=BEN)] == [0, 0, 0, 0]


def test_switching_off_goes_back_to_one_unnamed_set_and_deleting_a_member_drops_their_habits():
    h = book()
    route(h, [*one_day(10), identity(at(11, "08:00"), True, [ANA, BEN]), *stay(11, "09:00", "10:00", track=1, member=ANA),
              scene(at(11, "09:30"), caption="a plate of eggs"),
              identity(at(11, "12:00"), True, [ANA])])  # Ben is deleted
    assert list(h.members) == [ANA] and h.view() is h.members[ANA]
    route(h, [identity(at(11, "13:00"), False, [ANA]), *one_day(12), *one_day(13)])  # switched off on the 11th
    assert not h.attributing and h.view() is h.base and h.tags == {}
    assert [(c.day.day, c.value, c.applies) for c in h.base.history["meals"]] == [(10, 2, True), (11, 0, False), (12, 2, True)]
    assert [c.streak for c in h.base.history["meals"]] == [1, 1, 2]  # the day people were told apart neither helps nor hurts
    assert [(c.day.day, c.value, c.applies) for c in h.members[ANA].history["meals"]][-2:] == [(11, 1, False), (12, 0, False)]


# --- replay ---


def wire(bus: EventBus, h: MemberHabits) -> None:
    """The gateway's subscriptions (contexts/gateway/bootstrap.py), plus the two for recognition."""
    bus.subscribe("perception.zone", h.on_zone_event, replay=True)
    bus.subscribe("perception.scene", h.on_scene, replay=True)
    bus.subscribe("connector.item.synced", h.on_item, replay=True)
    bus.subscribe("impact.item.disposed", h.on_disposed, replay=True)
    bus.subscribe("habits.day.closed", h.on_day_closed, replay=True)
    bus.subscribe("habits.nudge.raised", h.on_nudge, replay=True)
    bus.subscribe("perception.identity", h.on_identity, replay=True)
    bus.subscribe("perception.person", h.on_person, replay=True)


def picture(h: MemberHabits, now) -> dict:
    views = {"": h.base, **h.members}
    return {m: ([t.digest(day(d), now) for d in (11, 12, 13, 14)], [t.history_for(i, 30) for i in IDS],
                {k: (a.desk_min, a.started_at) for k, a in t.assignments.items()})
            for m, t in views.items()} | {"tags": dict(h.tags), "attributing": h.attributing}


def test_a_rebuilt_tracker_gives_the_same_per_member_numbers_from_the_same_log(tmp_path):
    story = [
        *one_day(10), assignment(at(10, "09:00"), "COP4533 Milestone 1", at(12, "23:59")),
        identity(at(10, "20:00"), True, [ANA]),
        *stay(10, "21:00", "23:00", track=1, member=ANA),
        identity(at(11, "07:00"), True, [ANA, BEN]),  # Ben enrols the next morning
        *stay(11, "08:30", "12:00", track=2, member=ANA), scene(at(11, "11:00"), caption="a bowl of ramen"),
        ("tick", at(11, "12:05")),
        *stay(11, "12:30", "13:30", track=3, member=BEN), scene(at(11, "13:00"), objects=["red bull can", "plate"], zone_id="desk"),
        *stay(11, "14:00", "15:00", track=4), scene(at(11, "14:30"), caption="a burger"),  # a guest
        disposed(at(11, "15:10"), True),
        *stay(11, "23:50", "23:55", track=5, member=ANA)[:-1], ("tick", at(12, "00:05")),  # the ticker closes the 11th
        zone(at(12, "00:30"), "dwell", track=5, dwell_s=2400), zone(at(12, "01:00"), "exit", track=5),
        lost(at(12, "01:00")), *stay(12, "10:00", "11:00", track=1, member=BEN),  # ids start again after a restart
        calendar(at(12, "11:00"), "Study group", at(12, "13:00")), scene(at(12, "10:30"), objects=["keys"], zone_id="desk"),
        tagged(at(12, "12:40"), 1, BEN, camera="cam-door"), zone(at(12, "12:45"), "exit", "door", track=1),
        identity(at(12, "18:00"), False, [ANA, BEN]), *sit(12, "19:00", "20:00"), *one_day(13),
        identity(at(13, "20:00"), True, [ANA, BEN]), *stay(13, "21:00", "22:00", track=7, member=ANA),
        *stay(14, "09:00", "09:30", track=8, member=BEN), ("tick", at(15, "00:05")),
    ]

    async def live() -> MemberHabits:
        bus, h = EventBus(tmp_path), book(nudges=True)
        wire(bus, h)
        for e in story:
            await bus.publish_all(h.flush(e[1]) if isinstance(e, tuple) else [e])
        return h

    async def rebuilt() -> MemberHabits:
        bus, h = EventBus(tmp_path), book(nudges=True)
        wire(bus, h)
        await bus.replay_into()
        return h

    first, again = asyncio.run(live()), asyncio.run(rebuilt())
    now = at(15, "00:05")
    assert picture(again, now) == picture(first, now)
    # ... and the numbers are real ones, not two empty pictures.
    assert set(first.members) == {ANA, BEN} and first.attributing
    assert [c.day.day for c in first.members[ANA].history["meals"]] == [10, 11, 12, 13, 14]
    assert first.members[ANA].history["meals"][1].value == 1 and first.members[BEN].history["energy_drinks"][1].value == 1
    assert first.members[ANA].history["focus"][1].value == 3.5  # 08:30 to 12:00
    assert first.members[BEN].history["leave_ready"][2].value == 1  # the keys, on the 12th
    assert first.base.history["sort_right"][1].value == 100  # the can stays with the room
    log = "".join(p.read_text() for p in tmp_path.glob("*.jsonl"))
    assert "habits.signal.recorded" in log  # from the days with nobody enrolled or recognition off


# --- the API ---


def test_the_api_picks_a_member_and_is_unchanged_while_recognition_is_off():
    plain = TestClient(_app(tracker()))
    assert set(plain.get("/api/habits/today").json()) == {"day", "habits", "latest_nudge"}
    assert plain.get("/api/habits/today?member=m-ana").status_code == 404

    h = book()
    c = TestClient(_app(h))
    assert set(c.get("/api/habits/today").json()) == {"day", "habits", "latest_nudge"}
    assert c.get("/api/habits/today?member=m-ana").status_code == 404  # off: nobody to pick

    route(h, [identity(at(11, "08:00"), True, [ANA, BEN]), *stay(11, "09:00", "10:00", track=1, member=ANA),
              scene(at(11, "09:30"), caption="a plate of eggs")])
    meals = lambda body: next(r["value"] for r in body["habits"] if r["id"] == "meals")  # noqa: E731
    ana, ben, desk = (c.get(f"/api/habits/today{q}").json() for q in ("?member=m-ana", "?member=m-ben", ""))
    assert (meals(ana), ana["member"], ana["members"]) == (1, ANA, [ANA, BEN])
    assert (meals(ben), ben["member"]) == (0, BEN) and (meals(desk), desk["member"]) == (0, None)
    assert c.get("/api/habits/today?member=m-eve").status_code == 404
    assert [r["value"] for r in c.get("/api/habits/meals/history?member=m-ana").json()] == [1]
    assert c.get("/api/habits/meals/history?member=m-eve").status_code == 404

    route(h, [identity(at(11, "11:00"), True, [ANA])])
    assert c.get("/api/habits/today").json()["member"] == ANA  # one person enrolled: theirs without asking


def _app(tracker_or_book) -> FastAPI:
    app = FastAPI()
    app.include_router(habits_router(tracker_or_book), prefix="/api")
    return app
