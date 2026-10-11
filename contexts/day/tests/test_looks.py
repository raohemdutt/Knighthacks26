"""The look memory (stage one): what is remembered, what counts as similar, what is said, and that it stays on the box.

Every value below is invented for the test: nothing here was measured on anybody."""

import asyncio
import json
import logging
import re
from dataclasses import asdict, fields
from datetime import date, datetime, timedelta
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from bus import EventBus

from contexts.day.api import day_router
from contexts.day.application.loop import checkin_push
from contexts.day.domain import checkin as rule
from contexts.day.domain import looks as memory
from contexts.day.domain.checkin import Look
from contexts.day.domain.looks import Context, Moment, Outcome
from contexts.day.domain.narration import BOUNDS, DIAGNOSIS, check, render
from contexts.day.tests.test_checkin import NOW, TODAY, afternoon, at, bottle, box, of, person
from contexts.day.tests.test_day import TZ, places, zone  # noqa: F401
from contracts.day import BOX_ONLY_EVENTS, CheckinSuggestion, DayCheckinRaised, DayLookForgotten, DayNarrated
from contracts.perception import ExpressionObserved, IdentityChanged, ObservationRecorded

MEMBER = "m-ana"  # the member test_checkin's afternoon reads
FRIDAY = 9  # October 9, 2026: four days before the Tuesday of the check-in


# ---------- the rule (pure) ----------


def ctx(hour: int = 15, minute: int = 0, **more) -> Context:
    return Context(minute_of_day=hour * 60 + minute, weekday=more.pop("weekday", 4), **more)


def moment(day: int, hour: int = 15, member: str = MEMBER, outcome: Outcome | None = None, **context) -> Moment:
    when = at(hour, 0, day)
    return Moment(id=memory.moment_id(member, when, TZ), member_id=member, at=when, until=when + timedelta(minutes=6), day=when.date(),
                  context=ctx(hour, weekday=when.weekday(), **context), outcome=outcome or Outcome())


def test_similar_is_the_same_look_on_another_day_close_on_the_context():
    now = ctx(15, 40, sitting_min=100, since_drink_min=95, activity="study")
    close = moment(FRIDAY, sitting_min=120, since_drink_min=100, activity="study")
    assert memory.distance(now, close.context) < 0.2 and memory.similar(now, TODAY, MEMBER, [close]) is close
    # the parts: minutes at the desk, minutes since a drink, what they were doing, the time of day, each 0 to 1
    assert memory.distance(ctx(15, sitting_min=60), ctx(15, sitting_min=120)) == 0.5  # an hour apart at the desk: that part is 1
    assert memory.distance(ctx(15, sitting_min=60), ctx(15, sitting_min=300)) == 0.5  # ... and it gets no further than 1
    assert memory.distance(ctx(9, activity="study"), ctx(15, activity="class")) == 1.0
    assert memory.distance(ctx(23, 30, sitting_min=10), ctx(0, 30, sitting_min=10)) == 60 / memory.CLOCK_MIN / 2  # midnight is not a day apart


def test_nothing_similar_enough_is_never_forced():
    now = ctx(15, 40, sitting_min=100, since_drink_min=95, activity="study")
    morning = moment(FRIDAY, 9, sitting_min=10, activity="class")  # another kind of moment altogether
    assert memory.distance(now, morning.context) > memory.SIMILAR_MAX and memory.similar(now, TODAY, MEMBER, [morning]) is None
    assert memory.similar(now, TODAY, MEMBER, []) is None
    assert memory.similar(now, TODAY, MEMBER, [moment(13, sitting_min=100, since_drink_min=95)]) is None  # today is not "earlier"
    assert memory.similar(now, TODAY, MEMBER, [moment(FRIDAY, member="m-ben", sitting_min=100, since_drink_min=95)]) is None  # someone else's
    old = Moment(id="look-old", member_id=MEMBER, at=at(15, 0, 13) - timedelta(days=memory.CALLBACK_DAYS + 1), until=at(15, 0, 13),
                 day=TODAY - timedelta(days=memory.CALLBACK_DAYS + 1), context=now)
    assert memory.similar(now, TODAY, MEMBER, [old]) is None  # further back than the window
    # only the time of day in common: too little to call it the same kind of moment
    assert memory.distance(ctx(15), ctx(15)) is None and memory.similar(ctx(15), TODAY, MEMBER, [moment(FRIDAY)]) is None
    assert memory.distance(ctx(15, sitting_min=100), ctx(15, activity="study")) is None and memory.MIN_SHARED == 2  # nothing in common but the hour


def test_among_near_ties_the_most_recent_wins_and_a_clearly_closer_one_beats_a_recent_one():
    now = ctx(15, 40, sitting_min=100, since_drink_min=95)
    older, newer = moment(6, sitting_min=100, since_drink_min=95), moment(FRIDAY, sitting_min=110, since_drink_min=95)
    assert abs(memory.distance(now, older.context) - memory.distance(now, newer.context)) <= memory.NEAR_TIE
    assert memory.similar(now, TODAY, MEMBER, [older, newer]) is newer
    far = moment(12, sitting_min=150, since_drink_min=10)  # yesterday, but a different afternoon
    assert memory.distance(now, far.context) - memory.distance(now, older.context) > memory.NEAR_TIE
    assert memory.similar(now, TODAY, MEMBER, [older, far]) is older


def test_the_day_is_said_the_way_a_person_says_it():
    def when(day: int, hour: int = 15, month: int = 10) -> str:
        return memory.when_words(datetime(2026, month, day, hour, 0, tzinfo=TZ), TODAY, TZ)

    assert when(12) == "yesterday afternoon" and when(12, 22) == "last night" and when(12, 9) == "yesterday morning"
    assert when(9) == "Friday afternoon" and when(7, 19) == "Wednesday evening" and when(8, 23) == "Thursday night"
    assert when(6) == "last Tuesday afternoon"  # a week ago to the day
    assert when(2) == "the Friday before last" and when(29, month=9) == "about two weeks ago" and when(23, month=9) == "about three weeks ago"
    assert when(10, 1) == "Friday night"  # one in the morning belongs to the night before
    assert memory.short_day(at(15, 0, 12), TODAY, TZ) == "yesterday" and memory.short_day(at(15, 0, 9), TODAY, TZ) == "Fri"
    assert memory.short_day(at(15, 0, 2), TODAY, TZ) == "Oct 2"


def test_the_callback_names_the_one_most_telling_fact():
    def said(**context) -> str:
        return memory.callback(moment(FRIDAY, **context), TODAY, TZ)

    assert said(sitting_min=120, since_drink_min=100, doing="studying") == (
        "That's the same look you had Friday afternoon, when you'd gone two hours without moving.")
    assert said(sitting_min=65, since_drink_min=200) == "That's the same look you had Friday afternoon, when you hadn't picked up a drink in about three and a half hours."
    assert said(sitting_min=20, energy_drinks=3, energy_limit=2) == "That's the same look you had Friday afternoon, when you'd had three energy drinks."
    assert said(sitting_min=20, deadline="Project Milestone 1", deadline_hours=20) == (
        "That's the same look you had Friday afternoon, when Project Milestone 1 was due the next day.")
    assert said(sitting_min=20, doing="in a lecture") == "That's the same look you had Friday afternoon, when you were in a lecture."
    assert said() == "That's the same look you had Friday afternoon."  # nothing telling: only when it was


def test_the_outcome_is_what_the_box_saw_in_the_half_hour_after():
    t = at(15, 0, FRIDAY)

    def looks(start: datetime, n: int, share: float) -> list[Look]:
        return [Look(start + timedelta(minutes=3 * i), start + timedelta(minutes=3 * i + 3), 30, share * 30) for i in range(n)]

    sat, back = (at(13, 0, FRIDAY), t + timedelta(minutes=5)), (t + timedelta(minutes=15), t + timedelta(minutes=40))
    got = memory.outcome(t, t + timedelta(minutes=31), [sat, back], [], True, looks(t + timedelta(minutes=15), 5, 0.1))
    assert got == Outcome(known=True, got_up=True, away_min=10, drink=False, eased=True)
    # stayed put and the looks stayed strained
    stayed = memory.outcome(t, t + timedelta(minutes=31), [(at(13, 0, FRIDAY), t + timedelta(minutes=31))], [], True, looks(t, 10, 0.7))
    assert stayed == Outcome(known=True, got_up=False, drink=False, eased=False)
    # a drink was seen; a sip after the half hour is not the outcome
    assert memory.outcome(t, t + timedelta(hours=1), [], [t + timedelta(minutes=4)], True, []).drink is True
    assert memory.outcome(t, t + timedelta(hours=1), [], [t + timedelta(minutes=40)], True, []).drink is False
    # what the box could not see is unknown, never "no": nobody at the desk, no container in view, no looks after
    blind = memory.outcome(t, t + timedelta(hours=1), [], [], False, looks(t, 2, 0.7))
    assert blind == Outcome(known=True) and (blind.got_up, blind.drink, blind.eased) == (None, None, None)
    # left for under three minutes: not "got up"; left after a quarter of an hour: not "got up" either
    blink = memory.outcome(t, t + timedelta(minutes=31), [(at(13, 0, FRIDAY), t + timedelta(minutes=5)), (t + timedelta(minutes=7), t + timedelta(minutes=40))], [], False, [])
    late = memory.outcome(t, t + timedelta(minutes=31), [(at(13, 0, FRIDAY), t + timedelta(minutes=20))], [], False, [])
    assert blink.got_up is False and late.got_up is False
    assert memory.outcome(t, t + timedelta(minutes=29), [sat], [], True, []) == Outcome()  # too soon: not known yet


def test_what_it_learned_is_said_only_when_the_outcome_is_known_and_the_advice_set_is_kept():
    def of_(**o) -> tuple[str, str | None]:
        return memory.learned(moment(FRIDAY, outcome=Outcome(known=True, **o)))

    assert of_(got_up=True, away_min=10, eased=True) == ("Ten minutes away from the desk helped then.", "walk")
    assert of_(got_up=True, eased=True) == ("Getting up helped then.", "walk")
    assert of_(got_up=False, drink=True, eased=True) == ("A drink helped then.", "drink")
    assert of_(got_up=False, drink=False, eased=False) == ("You pushed through then, and it was still there half an hour later.", None)
    assert of_(got_up=True, eased=False) == ("", None) and of_(got_up=False, eased=True) == ("", None)  # nothing to claim
    assert of_(got_up=True) == ("", None) and memory.learned(moment(FRIDAY)) == ("", None)  # unknown: nothing is said
    todo = [CheckinSuggestion(kind="drink", text="Drink some water"), CheckinSuggestion(kind="walk", text="Take ten minutes outside"),
            CheckinSuggestion(kind="move_block", text="Move the study block to after dinner")]
    assert [s.kind for s in memory.rerank(todo, "walk")] == ["walk", "drink", "move_block"]
    assert [s.kind for s in memory.rerank(todo, "drink")] == ["drink", "walk", "move_block"] == [s.kind for s in memory.rerank(todo, None)]
    assert sorted(s.text for s in memory.rerank(todo, "walk")) == sorted(s.text for s in todo)  # the same advice, the same words
    rest = [CheckinSuggestion(kind="break", text="Take ten minutes away from the desk")]
    assert memory.rerank(rest, "walk") == rest


def test_no_sentence_it_can_build_uses_a_diagnosis_word():
    lines = []
    for o in (Outcome(), Outcome(known=True), Outcome(known=True, got_up=True, away_min=10, eased=True), Outcome(known=True, drink=True, eased=True),
              Outcome(known=True, eased=True), Outcome(known=True, got_up=True, eased=False), Outcome(known=True, got_up=False, drink=False, eased=False)):
        m = moment(FRIDAY, outcome=o, sitting_min=120, since_drink_min=100, energy_drinks=3, energy_limit=2, rhr_up=True, short_night=True,
                   doing="studying", deadline="Project Milestone 1", deadline_hours=20)
        lines += [memory.callback(m, TODAY, TZ), memory.learned(m)[0], memory.going_on(m.context), memory.went(m), memory.chip(m, TODAY, TZ),
                  memory.said_last(m, TODAY, TZ), memory.said_why(m, m, TODAY, TZ), memory.said_helped(m, TODAY, TZ)]
    lines += [memory.said_last(None, TODAY, TZ), memory.said_why(None, None, TODAY, TZ), memory.said_helped(None, TODAY, TZ)]
    assert not [w for line in lines for w in rule.diagnosis_in(line)]
    assert not [line for line in lines if re.search(r"look-|frames:|\d{2}:\d{2}|\bbpm\b", line)]  # no ids, no ranges, no heart value


def test_a_moment_holds_labels_and_numbers_only():
    names = {f.name for f in fields(Context)} | {f.name for f in fields(Outcome)} | {f.name for f in fields(Moment)}
    assert not [n for n in names if re.search(r"frame|picture|image|photo|vector|face|bpm|hrv|heart_rate|resting_hr", n)]
    assert str({f.name: f.type for f in fields(Context)}["rhr_up"]) == "bool | None"  # the heart rate: above the usual or not, never the value
    example = memory.labelled_examples([moment(FRIDAY, sitting_min=120, outcome=Outcome(known=True, got_up=True, eased=True))])
    assert [sorted(x) for x in example] == [["advice", "features", "look", "outcome"]] and "m-ana" not in json.dumps(example)


# ---------- the service, on a bus ----------


def scanned(*members: str, ts: datetime | None = None) -> IdentityChanged:
    return IdentityChanged(ts=ts or at(8, 0, 5), enabled=True, members=list(members) or [MEMBER], expressions=list(members))


def windows(start: datetime, n: int, share: float, member: str | None = MEMBER, camera: str = "cam-desk") -> list[ExpressionObserved]:
    out = []
    for i in range(n):
        a = start + timedelta(minutes=3 * i)
        out.append(ExpressionObserved(event_id=f"look-{camera}-{member}-{int(a.timestamp())}", ts=a + timedelta(minutes=3), camera_id=camera,
                                      member_id=member, window_start=a, window_end=a + timedelta(minutes=3), dominant="sad" if share >= 0.5 else "neutral",
                                      shares={"sad": share, "neutral": 1 - share}, samples=30, confidence=0.8))
    return out


def seen(ts: datetime, y: float) -> ObservationRecorded:
    ms = int(ts.timestamp() * 1000)
    return ObservationRecorded(event_id=f"obs-{ms}", ts=ts, camera_id="cam-desk", tracks=[person(), bottle(y)], frame_ref=f"frames:cam-desk:{ms}")


def friday(got_up: bool = True, member: str | None = MEMBER) -> list:
    """An earlier afternoon: sat down at 1, a drink at twenty past, strained from quarter to three (so the look holds
    at 3, two hours in). Then either up at five past three for ten minutes and calmer after, or put and still strained."""
    s, d = timedelta(seconds=1), FRIDAY
    events = [zone(at(13, 0, d), "desk"), *(seen(at(13, 0, d) + s * k, 0.70) for k in (1, 2, 3)), seen(at(13, 20, d), 0.50),
              seen(at(13, 20, d) + 2 * s, 0.50), seen(at(13, 21, d), 0.70), seen(at(14, 50, d), 0.70), *windows(at(14, 45, d), 6, 0.7, member)]
    if got_up:
        events += [zone(at(15, 5, d), "desk", "exit"), zone(at(15, 15, d), "desk"), *windows(at(15, 15, d), 6, 0.1, member),
                   zone(at(15, 40, d), "desk", "exit")]
    else:
        events += [*windows(at(15, 3, d), 9, 0.7, member), zone(at(15, 40, d), "desk", "exit")]
    return sorted(events, key=lambda e: e.ts)


SAID = ("You slept about five hours, your resting heart rate is up, and you've looked strained for the last twenty minutes. "
        "That's the same look you had Friday afternoon, when you'd gone two hours without moving. "
        "Ten minutes away from the desk helped then. "
        "You've been at the desk for about an hour and three quarters straight, and I haven't seen you pick up a drink since five past two. "
        "Take ten minutes outside, drink some water, and I can move the study block to after dinner if you say yes.")


async def tuesday(places, tmp_path, earlier: list, name: str = "a"):  # noqa: F811
    """A box whose log already holds `earlier` (written to the log and replayed, the way a box that has been running
    has it), then the Tuesday afternoon of test_checkin lived through it: the plan, the afternoon, the look at the rule."""
    await EventBus(tmp_path / "events").publish_all(earlier)
    loop, bus, out, ntfy = box(places, tmp_path, name)
    await bus.replay_into()
    await bus.publish_all(loop.planner.plan(at(8, 10)))
    loop.checkin.enabled = False
    for e in afternoon():
        await bus.publish(e)
    loop.checkin.enabled = True
    await bus.publish_all(loop.tick(NOW))
    await bus.publish_all(loop.tick(NOW + timedelta(seconds=90)))  # the coach spoke at 3:39:30: the check-in's words wait for the arbiter's gap
    return loop, bus, out, ntfy


def test_a_sustained_look_is_remembered_once_with_what_was_going_on_and_what_happened_next(places, tmp_path):  # noqa: F811
    loop, bus, out, _ = box(places, tmp_path)
    asyncio.run(bus.publish_all([scanned(MEMBER), *friday()]))
    [m] = loop.looks.remembered(at(16, 0, FRIDAY))  # one stretch of six windows: ONE moment
    assert m.id == "look-m-ana-20261009-1500" and m.at == at(15, 0, FRIDAY) and m.look == "strained"
    assert m.until == at(15, 18, FRIDAY)  # the rule's twenty minutes still held when they sat back down; by 3:21 they no longer did
    assert m.context == Context(minute_of_day=15 * 60, weekday=4, sitting_min=120, since_drink_min=99)  # no plan, no watch that day: unknown, not zero
    assert m.outcome == Outcome(known=True, got_up=True, away_min=10, drink=False, eased=True)
    assert of(out, DayCheckinRaised) == [] and not [e for e in out if e.type.startswith("day.look")]  # a projection: no event of its own
    shown = loop.looks.view(at(16, 0, FRIDAY))["moments"][0]
    assert shown["when"] == "This afternoon" and shown["going_on"] == "At the desk for two hours, no drink seen for about an hour and three quarters."
    assert shown["helped"] == "Ten minutes away from the desk helped: the look eased within half an hour."


def test_with_no_record_from_the_desk_zone_the_members_own_windows_say_how_long_they_sat(places, tmp_path):  # noqa: F811
    """A box whose desk zone recorded nothing for that time (no zone drawn, the zone engine off): an unbroken run of the
    member's own windows is them in front of the desk camera, a gap of three minutes or more is them gone from it."""
    d = FRIDAY
    turned_away = ExpressionObserved(event_id="look-away", ts=at(13, 3, d), camera_id="cam-desk", member_id=MEMBER, window_start=at(13, 0, d),
                                     window_end=at(13, 3, d), dominant="neutral", shares={}, samples=0, confidence=0.0, looks=30, turned_away=1.0)
    looks = [turned_away, *windows(at(13, 3, d), 34, 0.1), *windows(at(14, 45, d), 7, 0.7), *windows(at(15, 16, d), 6, 0.1)]
    loop, bus, out, _ = box(places, tmp_path)
    asyncio.run(bus.publish_all([scanned(MEMBER), *looks]))
    [m] = loop.looks.remembered(at(16, 0, d))
    assert m.at == at(15, 0, d) and m.context.sitting_min == 120 and m.context.since_drink_min is None  # from 1 PM: the window with no counted look counts
    assert m.outcome == Outcome(known=True, got_up=True, away_min=10, drink=None, eased=True)  # gone from 3:06 to 3:16
    assert loop.looks.presence.closed == [] and loop.looks.presence.open == {}  # nothing came from the desk zone
    # a break of under three minutes in the windows is not a break: the run goes on
    loop, bus, out, _ = box(places, tmp_path / "b", "b")
    asyncio.run(bus.publish_all([scanned(MEMBER), *windows(at(13, 0, d), 20, 0.1), *windows(at(14, 2, d), 10, 0.1), *windows(at(14, 32, d), 7, 0.7)]))
    [m] = loop.looks.remembered(at(16, 0, d))
    assert m.context.sitting_min == 107 and m.outcome.got_up is True and m.outcome.away_min is None and m.outcome.eased is None  # left and did not come back


def test_the_check_in_calls_back_to_it_and_leads_with_what_helped(places, tmp_path):  # noqa: F811
    loop, bus, out, ntfy = asyncio.run(tuesday(places, tmp_path, [scanned(MEMBER), *friday()]))
    [e] = of(out, DayCheckinRaised)
    back = e.callback
    assert back.said == "That's the same look you had Friday afternoon, when you'd gone two hours without moving."
    assert back.learned == "Ten minutes away from the desk helped then." and back.lead == "walk" and back.day == date(2026, 10, FRIDAY)
    assert back.chip == "Looked strained · like Fri" and back.moment_id == "look-m-ana-20261009-1500"
    assert back.lines == ["Friday afternoon, around 3 PM. At the desk for two hours, no drink seen for about an hour and three quarters.",
                          "Ten minutes away from the desk helped: the look eased within half an hour."]
    assert [s.kind for s in e.suggestions][:2] == ["walk", "drink"]  # re-ranked by what worked; the set and its words are the rule's own
    assert sorted(s.text for s in e.suggestions if s.kind != "move_block") == ["Drink some water", "Take ten minutes outside"]
    # said through the narrator's own path and the arbiter: one `day.narrated` for the check-in, after the coach's line
    [told] = [n for n in of(out, DayNarrated) if n.moment == "checkin"]
    assert told.spoken == SAID and e.proposes_replan and told.ref == e.event_id and not told.merged_into
    assert back.said in e.text and back.said in told.display
    sentences, chars = BOUNDS["checkin"]
    assert len(re.split(r"(?<=[.!?])\s+", told.spoken)) <= sentences and len(told.spoken) <= chars
    assert rule.diagnosis_in(told.spoken) == [] and rule.diagnosis_in(e.text) == [] and not set(DIAGNOSIS) & set(re.findall(r"[a-z]+", back.model_dump_json().lower()))
    facts = rule.facts(e.signals, e.suggestions, e.history, TZ, callback=back)
    assert check(told.spoken, facts, render(facts).spoken) == []  # the grounding check a model's rewrite must pass holds for it
    # the moment now carries what Atlas suggested then
    today = next(m for m in loop.looks.remembered(NOW) if m.day == TODAY)
    assert today.checkin_id == e.event_id and today.suggested[:2] == ("walk", "drink")
    # the card: the chip opens the remembered moment
    card = loop.checkin.card(e)
    assert card["callback"]["chip"] == "Looked strained · like Fri" and card["callback"]["lines"] == back.lines


def test_staying_put_last_time_is_said_gently_and_changes_no_order(places, tmp_path):  # noqa: F811
    loop, bus, out, _ = asyncio.run(tuesday(places, tmp_path, [scanned(MEMBER), *friday(got_up=False)]))
    [e] = of(out, DayCheckinRaised)
    assert e.callback.learned == "You pushed through then, and it was still there half an hour later." and e.callback.lead is None
    assert [s.kind for s in e.suggestions][:2] == ["drink", "walk"]


def test_with_no_earlier_look_like_it_nothing_is_said_about_the_past(places, tmp_path):  # noqa: F811
    # nothing remembered at all
    loop, bus, out, _ = asyncio.run(tuesday(places, tmp_path, [scanned(MEMBER)]))
    [e] = of(out, DayCheckinRaised)
    assert e.callback is None and "same look" not in e.text and loop.checkin.card(e)["callback"] is None
    # an earlier look, but in another kind of moment: a morning, ten minutes after sitting down
    morning = [zone(at(8, 50, FRIDAY), "desk"), *windows(at(8, 45, FRIDAY), 6, 0.7), zone(at(9, 40, FRIDAY), "desk", "exit")]
    loop, bus, out, _ = asyncio.run(tuesday(places, tmp_path / "b", [scanned(MEMBER), *morning], name="b"))
    [e] = of(out, DayCheckinRaised)
    assert len(loop.looks.moments) == 2 and e.callback is None and "same look" not in e.text
    assert [s.kind for s in e.suggestions][:2] == ["drink", "walk"]  # the rule's own order


def test_a_restart_rebuilds_the_same_moments_and_says_nothing_again(places, tmp_path):  # noqa: F811
    loop, bus, out, ntfy = asyncio.run(tuesday(places, tmp_path, [scanned(MEMBER), *friday()]))
    before = [asdict(m) for m in loop.looks.remembered(NOW + timedelta(hours=1))]
    assert len(before) == 2

    async def restart():
        again, bus2, out2, ntfy2 = box(places, tmp_path, "b")
        await bus2.replay_into()
        await bus2.publish_all(again.tick(NOW + timedelta(minutes=5)))
        return again, out2, ntfy2

    again, out2, ntfy2 = asyncio.run(restart())
    assert [asdict(m) for m in again.looks.remembered(NOW + timedelta(hours=1))] == before
    assert again.looks.view(NOW)["moments"] == loop.looks.view(NOW)["moments"] and again.looks.examples() == loop.looks.examples()
    assert out2 == [] and ntfy2.sent == []


# ---------- privacy: who is remembered, and that it never leaves ----------


def test_only_a_scanned_member_with_expressions_on_is_remembered(places, tmp_path):  # noqa: F811
    loop, bus, out, _ = box(places, tmp_path)
    # the log says who scanned and switched it on: someone else's looks, and the anonymous room's, are not remembered
    asyncio.run(bus.publish_all([scanned("m-ben"), *friday()]))
    assert loop.looks.moments == {} and loop.looks.looks == {} and loop.looks.view(at(16, 0, FRIDAY))["moments"] == []
    loop, bus, out, _ = box(places, tmp_path / "anon", "anon")
    asyncio.run(bus.publish_all([scanned(MEMBER), *friday(member=None)]))
    assert loop.looks.moments == {} and loop.looks.said("last", NOW) == memory.NOTHING
    # nobody switched it on: nothing at all, and a question about it is not this memory's to answer
    loop, bus, out, _ = box(places, tmp_path / "off", "off")
    asyncio.run(bus.publish_all([scanned(), *friday()]))
    assert loop.looks.moments == {} and loop.looks.said("last", NOW) is None and loop.looks.view(NOW)["on"] is False


def test_switching_expressions_off_stops_remembering_and_clears_what_was_remembered(places, tmp_path):  # noqa: F811
    loop, bus, out, _ = box(places, tmp_path)
    asyncio.run(bus.publish_all([scanned(MEMBER), *friday()]))
    assert len(loop.looks.moments) == 1
    asyncio.run(bus.publish(scanned(ts=at(18, 0, FRIDAY))))  # switched off: still enrolled, no longer read
    assert loop.looks.moments == {} and loop.looks.looks == {}
    asyncio.run(bus.publish_all(windows(at(14, 45, 10), 8, 0.7)))  # looks that arrive anyway are not remembered
    assert loop.looks.moments == {} and loop.looks.looks == {}

    async def restart():
        again, bus2, _, _ = box(places, tmp_path, "b")
        await bus2.replay_into()
        return again

    assert asyncio.run(restart()).looks.moments == {}  # and a replay of the log comes out the same


def app_of(loop, bus, token: str | None = None) -> TestClient:
    from fastapi import Header, HTTPException

    def guard(authorization: str | None = Header(default=None)) -> None:
        if token and authorization != f"Bearer {token}":
            raise HTTPException(401, "missing or wrong token")

    app = FastAPI()
    app.include_router(day_router(loop, publish=bus.publish, guard=guard, clock=lambda: NOW))
    return TestClient(app)


def test_forgetting_one_or_all_goes_through_the_authorised_route_and_survives_a_replay(places, tmp_path):  # noqa: F811
    loop, bus, out, _ = asyncio.run(tuesday(places, tmp_path, [scanned(MEMBER), *friday()]))
    [raised] = of(out, DayCheckinRaised)
    c = app_of(loop, bus, token="secret")
    body = c.get("/api/day/looks").json()
    assert body["on"] and body["count"] == 2 and [m["day"] for m in body["moments"]] == ["2026-10-13", "2026-10-09"]
    then = body["moments"][1]
    assert then["when"] == "Friday afternoon" and then["helped"].startswith("Ten minutes away from the desk helped")
    assert c.delete(f"/api/day/looks/{then['id']}").status_code == 401 and c.delete("/api/day/looks").status_code == 401  # not without the token
    auth = {"Authorization": "Bearer secret"}
    assert c.delete("/api/day/looks/look-nobody", headers=auth).status_code == 404
    assert c.delete(f"/api/day/looks/{then['id']}", headers=auth).json() == {"forgotten": then["id"], "left": 1}
    [gone] = [e for e in out if isinstance(e, DayLookForgotten)]
    assert gone.type in BOX_ONLY_EVENTS and gone.moment_id == then["id"]
    assert [m["day"] for m in c.get("/api/day/looks").json()["moments"]] == ["2026-10-13"]
    assert loop.checkin.card(raised)["callback"] is None  # the card's chip goes with the forgotten moment
    assert loop.looks.said("last", NOW) == memory.NOTHING

    async def restart(name: str):
        again, bus2, _, _ = box(places, tmp_path, name)
        await bus2.replay_into()
        return again

    assert list(asyncio.run(restart("b")).looks.moments) == ["look-m-ana-20261013-1533"]
    assert c.delete("/api/day/looks", headers=auth).json() == {"forgotten": 1, "left": 0}  # forget all: one call
    assert c.get("/api/day/looks").json()["moments"] == [] and asyncio.run(restart("c")).looks.moments == {}
    assert {e.type for e in out if isinstance(e, DayLookForgotten)} == {"day.look.forgotten"}


def fields_of(loop, raised: DayCheckinRaised) -> list[str]:
    """Every word of the memory that would give it away if it showed up somewhere it must not."""
    back = raised.callback
    shown = loop.looks.view(NOW)["moments"]
    return [back.said, back.learned, back.chip, *back.lines, back.moment_id, "same look", "helped then", "without moving", "look-m-ana",
            *(m["going_on"] for m in shown), *(m["helped"] for m in shown), *(m["id"] for m in shown), "since_drink_min", "sitting_min", "rhr_up"]


def test_the_push_names_the_advice_and_nothing_of_the_memory(places, tmp_path):  # noqa: F811
    loop, bus, out, ntfy = asyncio.run(tuesday(places, tmp_path, [scanned(MEMBER), *friday()]))
    [e] = of(out, DayCheckinRaised)
    assert e.callback is not None
    sent = " ".join(f"{title} {body}" for title, body in ntfy.sent)
    assert "Check-in" in sent and checkin_push(e) in sent
    assert not [word for word in (*fields_of(loop, e), "Friday", "strained", "look") if word in sent]
    assert checkin_push(e).startswith("Atlas has a check-in for you. Take ten minutes outside, drink some water.")


def test_a_circle_share_built_while_a_memory_exists_has_none_of_it(places, tmp_path):  # noqa: F811
    from contexts.circle.application.sharing import HABIT_FIELDS
    from contexts.circle.tests.test_health_privacy import wire
    from contexts.habits.tests.helpers import at as on

    async def run():
        loop, bus, out, ntfy = box(places, tmp_path)
        habits, sharing = wire(tmp_path, bus)  # Habits and the opted-in Circle sharing on the Day loop's own bus
        await bus.publish_all([scanned(MEMBER), *friday()])
        await bus.publish_all(loop.planner.plan(at(8, 10)))
        loop.checkin.enabled = False
        await bus.publish_all(afternoon())
        loop.checkin.enabled = True
        await bus.publish_all(loop.tick(NOW))
        await bus.publish_all(loop.tick(NOW + timedelta(seconds=90)))
        [e] = of(out, DayCheckinRaised)
        assert e.callback is not None and len(loop.looks.moments) == 2
        await bus.publish_all(habits.flush(on(14, "00:05")))  # the day closes: this is everything that leaves for the Circle
        rows = sharing.outbox.pending()
        assert rows and {r.kind for r in rows} == {"habit_summary"} and all(set(r.payload) == HABIT_FIELDS for r in rows)
        leaving = json.dumps([r.payload for r in rows])
        assert not [word for word in (*fields_of(loop, e), "strained", "look", "Friday", "callback") if word in leaving]
        assert len(sharing.sync(on(14, "00:06"))) == len(rows)  # ... and what is pushed to the shared database is those rows

    asyncio.run(run())


def test_an_export_built_while_a_memory_exists_has_none_of_it(places, tmp_path):  # noqa: F811
    from contexts.agent.application.memory_port import FakeMemory
    from contexts.learning.application.export import DatasetExporter, teach
    from contexts.workspace.domain.approvals import Approvals
    from contracts.experience import ApprovalDecided
    from contracts.learning import TaskSpec
    from contracts.memory import Episode

    loop, bus, out, _ = asyncio.run(tuesday(places, tmp_path, [scanned(MEMBER), *friday()]))
    [e] = of(out, DayCheckinRaised)
    c = app_of(loop, bus)
    assert c.delete("/api/day/looks/look-m-ana-20261009-1500").status_code == 200  # a forgotten look is in the log too
    target = tmp_path / "datasets"
    target.mkdir()
    published = []
    episode = Episode(id="e1", camera_id="cam-desk", start=NOW - timedelta(hours=1), captions=["a person at the desk"], keyframe_refs=["frames:cam-desk:e1"])
    approvals = Approvals({"export_training_data": DatasetExporter(FakeMemory([episode]), lambda ref: b"jpeg", published.append, out_dir=target)})
    _, proposal = teach(TaskSpec(task="desk-objects", instruction="List small objects on the desk as JSON.", camera="cam-desk"), NOW)
    approvals.on_proposed(proposal)
    [done] = asyncio.run(approvals.on_decided(ApprovalDecided(ts=NOW, action_id=proposal.action_id, decision="approve")))
    assert done.status == "done", done.result
    written = b"".join(p.read_bytes() for p in target.rglob("*") if p.is_file()).decode(errors="replace") + published[0].model_dump_json()
    assert not [word for word in (*fields_of(loop, e), "strained", "Friday", "callback", "day.look") if word in written]


def test_nothing_of_it_is_logged_at_info_level(places, tmp_path, caplog):  # noqa: F811
    with caplog.at_level(logging.INFO):
        loop, bus, out, _ = asyncio.run(tuesday(places, tmp_path, [scanned(MEMBER), *friday()]))
        [e] = of(out, DayCheckinRaised)
        c = app_of(loop, bus)
        c.get("/api/day/looks")
        loop.looks.said("why", NOW), loop.looks.said("helped", NOW), loop.looks.examples()
        c.delete("/api/day/looks")
    logged = " ".join(r.getMessage() for r in caplog.records if r.levelno >= logging.INFO and r.name.startswith("atlas"))
    assert e.callback is not None and not [word for word in (*fields_of(loop, e), "same look", "Friday afternoon") if word in logged]


def test_what_the_screen_is_given_has_no_picture_and_no_heart_value(places, tmp_path):  # noqa: F811
    loop, bus, out, _ = asyncio.run(tuesday(places, tmp_path, [scanned(MEMBER), *friday()]))
    [e] = of(out, DayCheckinRaised)
    body = json.dumps(app_of(loop, bus).get("/api/day/looks").json()) + e.callback.model_dump_json() + json.dumps(loop.looks.examples())
    assert "frames:" not in body and "frame_ref" not in body and "bpm" not in body and "resting_hr" not in body
    today = next(m for m in loop.looks.remembered(NOW) if m.day == TODAY)
    assert today.context.rhr_up is True and today.context.short_night is True  # the watch, as two booleans
    night = loop.checkin.nights[TODAY]
    assert night.resting_hr == 62 and not re.search(r"\b62(?:\.0)?\b", body)  # the value itself is nowhere in it


def test_nothing_outside_the_day_loop_reads_the_look_memory():
    """Structural: the code that leaves the box (the Circle, the League's sync, the Learning lab's export, the phone
    push) names neither the look memory nor its event, so none of it can be handed a remembered look. The agent
    reaches it only through the port the composition root wires (`talk.look_memory`)."""
    root = Path(__file__).resolve().parents[3]
    for folder in ("contexts/circle", "contexts/play/league", "contexts/learning", "contexts/experience", "contexts/vault", "contexts/connectors", "lab"):
        for path in (root / folder).rglob("*.py"):
            if "tests" in path.parts or path.name.startswith("test_"):
                continue
            text = path.read_text(encoding="utf-8")
            assert not re.search(r"day\.look|LookMemory|domain\.looks|application\.looks|CheckinCallback|labelled_examples", text), path
    assert "day.look.forgotten" in BOX_ONLY_EVENTS
    agent = "".join(p.read_text(encoding="utf-8") for p in (root / "contexts/agent").rglob("*.py") if "tests" not in p.parts)
    assert not re.search(r"contexts\.day\.(?:application|domain)\.looks|LookMemory", agent)


# ---------- asked ----------


def test_the_three_questions_are_answered_from_the_memory_for_the_ear(places, tmp_path):  # noqa: F811
    loop, bus, out, _ = asyncio.run(tuesday(places, tmp_path, [scanned(MEMBER), *friday()]))
    ask = loop.looks.said
    assert ask("last", NOW) == "The last time you looked this strained was Friday afternoon, when you'd gone two hours without moving."
    assert ask("helped", NOW) == "Last time, Friday afternoon, ten minutes away from the desk helped: the look eased within half an hour."
    assert ask("why", NOW) == ("I can't know how you feel, only what I saw: your face has read as strained for a while, you'd been at the desk "
                               "for about ninety minutes, and you slept short. "
                               "That's the same look you had Friday afternoon, when you'd gone two hours without moving.")
    # a day later, with no look lately: it says so, and never guesses
    assert ask("why", NOW + timedelta(days=1)) == ("I don't think that right now: your face hasn't read as strained lately. "
                                                   "I can only go by how you look to the camera.")
    assert ask("last", NOW + timedelta(days=1)) == "The last time you looked this strained was yesterday afternoon, when you'd gone about ninety minutes without moving."
    assert ask("anything else", NOW) is None
