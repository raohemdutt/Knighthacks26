"""The check-in (D4): the rule (pure), the words, the service on a bus, replay, the push and the API.

Every health value below is invented for the test: nothing here was measured on anybody."""

import asyncio
from datetime import date, datetime, timedelta
from itertools import combinations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from bus import EventBus
from contexts.agent.application.memory_port import FakeMemory
from contexts.connectors.application.hub import ConnectorHub
from contexts.day.api import day_router
from contexts.day.application.loop import DayLoop, checkin_push, day_phone
from contexts.day.domain import checkin as rule
from contexts.day.domain.checkin import DrinkWatch, Gate, Look, Night, PastSession, Reading
from contexts.day.domain.narration import DIAGNOSIS, check, render
from contexts.day.tests.test_day import CLOCK, TZ, never, places, zone  # noqa: F401
from contexts.play.dayrun.models import Move
from contexts.play.dayrun.service import DayRun
from contracts.connectors import ItemSynced
from contracts.day import BOX_ONLY_EVENTS, DayCheckinAnswered, DayCheckinRaised, DayNarrated, DayReplanProposed, PlanMove
from contracts.memory import Item
from contracts.perception import ExpressionObserved, IdentityChanged, ObservationRecorded, SceneDescribed
from contracts.types import Box, Track

TODAY = date(2026, 10, 13)  # a Tuesday with one lecture in the morning and nothing fixed after it


def at(hour: int, minute: int = 0, day: int = 13) -> datetime:
    return datetime(2026, 10, day, hour, minute, tzinfo=TZ)


NOW = at(15, 40)


# ---------- invented inputs ----------


def nights(asleep: int | None = None, hr: float | None = None, before: int = 8, bad: tuple[int, ...] = ()) -> dict[date, Night]:
    """`before` ordinary nights (about 7 h 40 min, resting heart rate about 55), then today's. `bad`: days ago that were short and up."""
    out = {}
    for i in range(1, before + 1):
        day = TODAY - timedelta(days=i)
        out[day] = Night(day, 300 if i in bad else 450 + (i % 3) * 10, 63.0 if i in bad else 55.0 + i % 2, ref=f"health:sleep:{day}")
    if asleep is not None or hr is not None:
        out[TODAY] = Night(TODAY, asleep, hr, ref=f"health:sleep:{TODAY}")
    return out


def looks(share: float, minutes: int = 20, end: datetime = NOW, samples: int = 30) -> list[Look]:
    return [Look(end - timedelta(minutes=m + 3), end - timedelta(minutes=m), samples, share * samples, f"expr-{m}") for m in range(0, minutes, 3)][::-1]


def person(x: float = 0.5) -> Track:
    return Track(track_id=1, label="person", confidence=0.9, box=Box(x1=x - 0.15, y1=0.40, x2=x + 0.15, y2=0.90))


def bottle(y: float, x: float = 0.5, label: str = "bottle") -> Track:
    return Track(track_id=7, label=label, confidence=0.8, box=Box(x1=x - 0.03, y1=y - 0.04, x2=x + 0.03, y2=y + 0.04))


def drank(*lifts: datetime, seen: datetime | None = NOW, label: str = "bottle") -> DrinkWatch:
    """A bottle resting on the desk, picked up at each of `lifts`, and still in view at `seen`."""
    w = DrinkWatch()
    start = min(lifts, default=NOW - timedelta(hours=2)) - timedelta(minutes=5)
    for s in range(3):
        w.see(start + timedelta(seconds=s), [person(), bottle(0.70, label=label)], "frames:cam-desk:1")
    for t in lifts:
        for s in (0, 2):
            w.see(t + timedelta(seconds=s), [person(), bottle(0.50, label=label)], f"frames:cam-desk:{int(t.timestamp())}")
        w.see(t + timedelta(seconds=10), [person(), bottle(0.70, label=label)], "frames:cam-desk:2")
    if seen:
        w.see(seen, [person(), bottle(0.70, label=label)], "frames:cam-desk:3")
    return w


def reading(now: datetime = NOW, **kw) -> Reading:
    return Reading(now=now, tz=TZ, day=TODAY, **kw)


SAT = [(at(14, 0), NOW)]  # at the desk for 100 minutes
SHORT = {"nights": nights(asleep=300)}
UP = {"nights": nights(hr=62.0)}
STRAINED = {"looks": looks(0.6)}
SITTING = {"desk": SAT}
NO_DRINK = {"desk": SAT, "drinks": drank(at(14, 5))}


def found(**kw) -> set[str]:
    return rule.kinds(rule.signals(reading(**kw)))


# ---------- the person's own baseline ----------


def test_a_baseline_is_the_persons_own_median_and_needs_enough_nights():
    assert rule.baseline([1, 2, 3, 4]) is None  # fewer than five: nothing is said
    b = rule.baseline([450, 460, 470, 450, 460, None])
    assert (b.median, b.spread, b.n) == (460, 10, 5)
    mine = nights(before=8)
    assert rule.own(mine, TODAY, "asleep_min").median == 460 and rule.own(mine, TODAY, "resting_hr").median == 55.5
    far = {d - timedelta(days=30): n for d, n in mine.items()}  # nights a month back are not this fortnight's baseline
    assert rule.own(far, TODAY, "asleep_min") is None


def test_short_and_up_are_measured_against_that_baseline_never_a_fixed_number():
    assert rule.short_night(nights(asleep=400), TODAY) and not rule.short_night(nights(asleep=401), TODAY)  # an hour under their own 460
    assert not rule.short_night(nights(asleep=300, before=4), TODAY)  # no baseline yet: no claim, however short
    long_sleeper = {d: Night(d, 560) for d in nights()} | {TODAY: Night(TODAY, 480)}
    assert rule.short_night(long_sleeper, TODAY)  # eight hours is short for someone who sleeps over nine
    short_sleeper = {d: Night(d, 320) for d in nights()} | {TODAY: Night(TODAY, 300)}
    assert not rule.short_night(short_sleeper, TODAY)  # and five is ordinary for someone who always sleeps five
    assert rule.resting_up(nights(hr=58.5), TODAY) and not rule.resting_up(nights(hr=58.4), TODAY)  # 3 bpm over their own 55.5
    jumpy = {d: Night(d, 450, 50.0 + 6 * (i % 3)) for i, d in enumerate(nights())} | {TODAY: Night(TODAY, 450, 63.0)}
    assert rule.own(jumpy, TODAY, "resting_hr").spread == 6 and not rule.resting_up(jumpy, TODAY)  # inside their own spread
    assert not rule.resting_up(nights(asleep=300), TODAY)  # a night with no heart rate says nothing about it


# ---------- each signal, and what it takes ----------


def test_each_signal_is_there_only_with_its_evidence():
    assert found() == set()
    assert found(**SHORT) == {"sleep"} and found(**UP) == {"resting_hr"} and found(**STRAINED) == {"expression"}
    assert found(**SITTING) == {"sitting"} and found(**NO_DRINK) == {"sitting", "drink"}
    assert found(looks=looks(0.4)) == set()  # mostly not strained
    assert found(looks=looks(0.9, samples=2)) == set()  # too few looks to say
    assert found(looks=looks(0.9, minutes=9)) == set()  # nine minutes are not "the last twenty"
    assert found(looks=looks(0.9, end=NOW - timedelta(minutes=8))) == set()  # an old stretch
    assert found(desk=[(at(14, 45), NOW)]) == set()  # 55 minutes: not long yet
    assert found(desk=[(at(13, 0), at(15, 0))]) == set()  # left the desk 40 minutes ago: nobody is sitting
    assert found(desk=[(at(14, 0), at(14, 50)), (at(14, 52), NOW)]) == {"sitting"}  # two minutes up is not a break
    assert found(desk=[(at(14, 0), at(14, 50)), (at(14, 55), NOW)]) == set()  # five minutes is


def test_the_drink_clause_is_only_what_the_camera_can_know():
    assert found(desk=SAT, drinks=drank(at(14, 5))) == {"sitting", "drink"}  # 95 minutes since the last one seen
    assert found(desk=SAT, drinks=drank(at(14, 20))) == {"sitting"}  # 80 minutes: not long enough
    assert found(desk=SAT, drinks=DrinkWatch()) == {"sitting"}  # no container ever in view: nothing can be said
    assert found(desk=SAT, drinks=drank(at(14, 5), seen=None)) == {"sitting", "drink"}
    gone = drank(seen=at(11, 0))
    assert found(desk=SAT, drinks=gone) == {"sitting"}  # the bottle left the picture hours ago
    [since] = [s for s in rule.signals(reading(**NO_DRINK)) if s.kind == "drink"]
    assert since.since == at(14, 5) + timedelta(seconds=2) and since.summary == "No drink since 2:05 PM" and "bottle" in since.detail
    assert since.evidence_refs[0].startswith("frames:cam-desk:")
    never_lifted = drank()  # in view the whole time, never picked up
    [none] = [s for s in rule.signals(reading(desk=SAT, drinks=never_lifted)) if s.kind == "drink"]
    assert none.since is None and none.summary == "No drink this sitting" and "since you sat down at 2 PM" in none.detail
    # a drink before this sitting does not count for it: the claim is about the time the camera watched
    before = drank(at(12, 0))
    [s] = [s for s in rule.signals(reading(desk=SAT, drinks=before)) if s.kind == "drink"]
    assert s.since is None


def test_a_pick_up_is_a_container_raised_to_head_height_twice_and_a_caption_counts():
    w = DrinkWatch()
    t = at(14, 0)
    assert not w.see(t, [person(), bottle(0.50)])  # raised, but nobody knows where it rests yet
    for s in range(1, 4):
        assert not w.see(t + timedelta(seconds=s), [person(), bottle(0.70)])  # resting on the desk
    assert w.container[1] == "bottle" and w.moments == []
    assert not w.see(t + timedelta(seconds=10), [person(), bottle(0.50)])  # one raised look: a blink
    assert not w.see(t + timedelta(seconds=30), [person(), bottle(0.50)])  # the second came too late
    assert w.see(t + timedelta(seconds=32), [person(), bottle(0.50)], "frames:cam-desk:9")  # two looks: picked up
    assert not w.see(t + timedelta(seconds=34), [person(), bottle(0.48)])  # the same drink, not another
    assert w.moments == [(t + timedelta(seconds=32), "frames:cam-desk:9")]
    assert not w.see(t + timedelta(minutes=5), [bottle(0.50)]) and not w.see(t + timedelta(minutes=5, seconds=2), [bottle(0.50)])  # nobody there
    assert not w.see(t + timedelta(minutes=6), [person(x=0.2), bottle(0.50)])  # raised, but not in front of the person
    assert not w.see(t + timedelta(minutes=7), [person(), bottle(0.66)]) and not w.see(t + timedelta(minutes=7, seconds=2), [person(), bottle(0.66)])
    assert len(w.moments) == 1  # nudged along the desk, not lifted
    assert w.caption(at(15, 0), "a person drinking from a bottle at the desk", "frames:cam-desk:10")
    assert not w.caption(at(15, 5), "two energy drinks next to the laptop")  # a thing on the desk, not a sip
    assert [m[0] for m in w.moments] == [t + timedelta(seconds=32), at(15, 0)]
    w.listed(at(15, 30), ["keys", "Cup"], "frames:cam-desk:11")
    assert w.container == (at(15, 30), "cup", "frames:cam-desk:11")
    assert w.last(at(16, 0), at(14, 30))[0] == at(15, 0) and w.last(at(14, 30), at(14, 10)) is None


# ---------- when it speaks ----------


@pytest.mark.parametrize("alone", [SHORT, UP, STRAINED, SITTING, NO_DRINK])
def test_no_single_signal_speaks(alone):
    assert not rule.agree(rule.signals(reading(**alone)))


def test_two_independent_signals_speak_and_the_desk_counts_once():
    def speaks(**kw) -> bool:
        return rule.agree(rule.signals(reading(**kw)))

    both = {"nights": nights(asleep=300, hr=62.0)}
    assert not speaks(**both)  # two watch signals: nothing is happening right now
    assert not speaks(**NO_DRINK)  # a long sitting with no drink is one thing said twice
    assert speaks(**SHORT, **SITTING) and speaks(**UP, **STRAINED) and speaks(**STRAINED, **SITTING) and speaks(**SHORT, **STRAINED)
    assert speaks(**both, **NO_DRINK, **STRAINED)
    assert rule.agree([]) is False


def test_it_is_held_back_by_a_class_a_time_to_leave_an_empty_desk_the_night_and_the_cooldown():
    assert rule.held(NOW, Gate()) is None
    assert rule.held(NOW, Gate(awake=False)) == "outside waking hours"
    assert rule.held(NOW, Gate(at_desk=False)) == "nobody at the desk"
    assert rule.held(NOW, Gate(fixed="COP4533 lecture")) == "during COP4533 lecture"
    leave = Gate(leave_by=at(16, 5), event_start=at(16, 20))
    assert rule.held(NOW, leave) == "close to a time to leave"  # 25 minutes before leaving
    assert rule.held(at(15, 30), leave) is None  # 35 minutes before: still fine
    assert rule.held(at(16, 10), leave) == "close to a time to leave"  # it is time to go, not to check in
    assert rule.held(at(16, 30), leave) is None
    assert rule.held(NOW, Gate(last=at(14, 11))) == "a check-in was raised less than ninety minutes ago"
    assert rule.held(NOW, Gate(last=at(14, 10))) is None
    assert rule.held(at(13, 0), Gate(last=at(14, 11))) is not None  # an event that arrives late, with an older time
    assert rule.held(NOW, Gate(last=at(15, 0), at_desk=False)).startswith("a check-in was raised")  # the cooldown is said first


# ---------- the earlier days ----------


def fell(day_ago: int, at_hour: int = 15, desk: int = 20) -> list[PastSession]:
    d = TODAY - timedelta(days=day_ago)
    start = datetime(d.year, d.month, d.day, at_hour, 30, tzinfo=TZ)
    return [PastSession(start - timedelta(hours=5), 60, 55, f"note-{day_ago}-a"), PastSession(start, 90, desk, f"note-{day_ago}-b")]


def test_history_needs_three_days_like_this_one_that_all_fell_apart():
    mine = nights(asleep=300, hr=62.0, before=12, bad=(2, 4, 6, 9))
    sessions = {TODAY - timedelta(days=n): fell(n, 15 if n != 4 else 14) for n in (2, 4, 6)}
    now = rule.signals(reading(nights=mine, **STRAINED))
    past, chip = rule.history(reading(nights=mine, sessions=sessions), now)
    assert past.matches == 3 and past.by == "15:30" and past.outcome == "your study blocks fell apart by 3:30 PM"
    assert [d.day for d in past.days] == [TODAY - timedelta(days=n) for n in (2, 4, 6)]
    assert past.days[0].text == "20 of 90 planned minutes at the desk from 3:30 PM on"
    assert chip.kind == "history" and chip.summary == "3 days like this before" and chip.evidence_refs == ["note-2-b", "note-4-b", "note-6-b"]
    assert "a short night and resting heart rate up" in chip.detail

    def none(**kw) -> bool:
        return rule.history(reading(**({"nights": mine, "sessions": sessions} | kw)), now) is None

    assert none(sessions={d: s for d, s in list(sessions.items())[:2]})  # two days are not a pattern
    held = sessions | {TODAY - timedelta(days=4): fell(4, desk=80)}
    assert none(sessions=held)  # one of the three held up: no claim
    ordinary = sessions | {TODAY - timedelta(days=3): fell(3)}  # an ordinary night's bad afternoon is not "like this"
    assert rule.history(reading(nights=mine, sessions=ordinary), now)[0].matches == 3
    assert rule.history(reading(nights=mine, sessions=sessions), rule.signals(reading(**STRAINED, **SITTING))) is None  # no watch signal today
    only_short = nights(asleep=300, before=12, bad=(2, 4, 6))
    assert rule.history(reading(nights=only_short, sessions=sessions), rule.signals(reading(nights=only_short))) is not None
    unmeasured = {d: [] for d in sessions}
    assert none(sessions=unmeasured)


def test_a_day_fell_apart_from_the_block_after_which_nothing_held():
    a, b, c = (PastSession(at(h), 60, desk, str(h)) for h, desk in ((9, 50), (13, 10), (16, 20)))
    assert rule.fell_apart([a, b, c]) == b and rule.fell_apart([c, a, b]) == b
    assert rule.fell_apart([a, b, PastSession(at(16), 60, 45)]) is None  # the last block held
    assert rule.fell_apart([a]) is None and rule.fell_apart([]) is None


# ---------- what it says ----------

ALL = {"nights": nights(asleep=300, hr=62.0, before=12, bad=(2, 4, 6)), "looks": looks(0.6), "desk": SAT, "drinks": drank(at(14, 5)),
       "sessions": {TODAY - timedelta(days=n): fell(n) for n in (2, 4, 6)}}
MOVE = "Move the study block to after dinner"


def said(move: str | None = None, **kw):
    r = reading(**kw)
    now = rule.signals(r)
    past = rule.history(r, now)
    todo = rule.suggestions(now, move)
    facts = rule.facts(now + ([past[1]] if past else []), todo, past[0] if past else None, TZ)
    return render(facts), facts


def test_the_sentence_reads_like_the_approved_one_with_every_signal():
    words, facts = said(MOVE, **ALL)
    assert words.spoken == ("You slept about five hours, your resting heart rate is up, and you've looked strained for the last twenty "
                            "minutes. The last three times a day looked like this, your study blocks fell apart by half past three. "
                            "You've been at the desk for about an hour and three quarters straight, and I haven't seen you pick up a "
                            "drink since five past two. Drink some water, take ten minutes outside, and I can move the study block to "
                            "after dinner if you say yes.")
    assert words.display == ("You slept about 5 h, your resting heart rate is up, and you've looked strained for the last 20 minutes. "
                             "The last 3 times a day looked like this, your study blocks fell apart by 3:30 PM. You've been at the desk "
                             "for 100 minutes straight, and I haven't seen you pick up a drink since 2:05 PM. Drink some water, take ten "
                             "minutes outside, and I can move the study block to after dinner if you say yes.")
    assert "bpm" not in str(facts) and "62" not in str(facts) and "55" not in str(facts)  # a heart rate is only ever "up"


def test_two_three_and_four_signals_read_naturally():
    assert said(**SHORT, **SITTING)[0].spoken == ("You slept about five hours. You've been at the desk for about an hour and three quarters "
                                                  "straight. Take ten minutes outside.")
    assert said(MOVE, **UP, **STRAINED)[0].spoken == ("Your resting heart rate is up and you've looked strained for the last twenty minutes. "
                                                      "Take ten minutes away from the desk, and I can move the study block to after dinner "
                                                      "if you say yes.")
    three = {"nights": nights(asleep=330, hr=62.0), "looks": looks(0.6)}
    assert said(**three)[0].spoken == ("You slept about five and a half hours, your resting heart rate is up, and you've looked strained "
                                       "for the last twenty minutes. Take ten minutes away from the desk.")
    four = {"nights": nights(asleep=300, hr=62.0), "desk": SAT, "drinks": drank()}
    assert said(**four)[0].spoken == ("You slept about five hours and your resting heart rate is up. You've been at the desk for about an "
                                      "hour and three quarters straight, and I haven't seen you pick up a drink in that time. Drink some "
                                      "water and take ten minutes outside.")


CLAUSES = {"sleep": "slept", "resting_hr": "resting heart rate", "expression": "looked strained", "sitting": "at the desk for",
           "drink": "pick up a drink"}
PARTS = {"sleep": {"nights": nights(asleep=300)}, "resting_hr": {"nights": nights(hr=62.0)}, "expression": STRAINED, "sitting": SITTING,
         "drink": {"drinks": drank(at(14, 5))}}


@pytest.mark.parametrize("have", [c for n in range(0, 6) for c in combinations(CLAUSES, n)])
def test_a_clause_is_said_only_when_its_signal_is_there_and_never_as_a_diagnosis(have):
    kw: dict = {}
    for kind in have:
        kw |= PARTS[kind]
    if "sleep" in have and "resting_hr" in have:
        kw["nights"] = nights(asleep=300, hr=62.0)
    if "drink" in have:
        kw["desk"] = SAT  # no drink clause without somebody sitting there
    present = rule.kinds(rule.signals(reading(**kw)))
    words, facts = said(MOVE if have else None, **kw)
    for kind, clause in CLAUSES.items():
        assert (clause in words.spoken) == (kind in present) and (clause in words.display) == (kind in present)
    assert "times a day looked like this" not in words.spoken  # no earlier days in these inputs
    assert ("Drink some water" in words.display) == ("drink" in present)  # the nudge follows its signal
    for text in (words.spoken, words.display, *(s.detail + " " + s.summary for s in rule.signals(reading(**kw)))):
        assert rule.diagnosis_in(text) == []
    assert check(words.spoken, facts, words.spoken) == []  # the template is grounded in its own facts


def test_no_diagnosis_word_anywhere_and_a_rewrite_that_adds_one_is_rejected():
    words, facts = said(MOVE, **ALL)
    assert {"stressed", "sick", "dehydrated", "tired"} <= DIAGNOSIS
    for text in (words.spoken, words.display, " ".join(rule.LIMITS).replace("not a diagnosis", "").replace("is a diagnosis", "")):
        assert rule.diagnosis_in(text) == []
    better = words.spoken.replace("You slept about five hours,", "You only slept about five hours,")
    assert check(better, facts, words.spoken) == []  # a faithful rewrite passes
    assert any("diagnosis" in p for p in check(words.spoken.replace("looked strained", "looked stressed"), facts, words.spoken))
    assert any("diagnosis" in p for p in check(words.spoken + " You may be dehydrated.", facts, words.spoken))
    assert check(words.spoken.replace("five hours", "four hours"), facts, words.spoken)  # a changed number
    assert check(words.spoken.replace("five past two", "three o'clock"), facts, words.spoken)  # a changed time
    assert check(words.spoken.replace("after dinner", "after lunch"), facts, words.spoken)  # a changed plan
    assert check(words.spoken.replace("three times", "seven times"), facts, words.spoken)  # a changed count


def test_suggestions_follow_the_signals():
    def kinds(move=None, **kw):
        return [(s.kind, s.text) for s in rule.suggestions(rule.signals(reading(**kw)), move)]

    assert kinds(**SHORT, **STRAINED) == [("break", "Take ten minutes away from the desk")]
    assert kinds(**SHORT, **SITTING) == [("walk", "Take ten minutes outside")]
    assert kinds(MOVE, **SHORT, **NO_DRINK) == [("drink", "Drink some water"), ("walk", "Take ten minutes outside"), ("move_block", MOVE)]


# ---------- "move the block" ----------


def moves_of(*rows) -> list[Move]:
    return [Move(slot="s1", kind=kind, minutes=minutes, start=start) for kind, start, minutes in rows]


def test_the_blocks_before_the_next_meal_move_to_straight_after_it():
    plan = moves_of(("eat", at(15, 31), 30), ("study", at(16, 2), 60), ("rest", at(17, 2), 15), ("study", at(17, 17), 45),
                    ("eat", at(18, 30), 30), ("study", at(19, 1), 90), ("sleep", at(23, 30), 480), ("study", at(8, 30, 14), 60))
    out, moved, meal = rule.after_meal(plan, NOW, TZ)
    assert [(m.kind, m.minutes) for m in out] == [("eat", 30), ("rest", 15), ("eat", 30), ("study", 60), ("study", 45), ("study", 90),
                                                 ("sleep", 480), ("study", 60)]
    assert moved == [3, 4] and meal == 2 and out[3].start == at(19, 1) and out[4].start is None
    assert rule.after_meal(moves_of(("eat", at(18, 30), 30), ("study", at(19, 1), 90)), NOW, TZ) is None  # already after dinner
    assert rule.after_meal(moves_of(("study", at(16, 0), 60), ("sleep", at(23, 30), 480)), NOW, TZ) is None  # no meal left today
    assert rule.after_meal(moves_of(("study", at(18, 0), 30), ("eat", at(18, 30), 30)), NOW, TZ) is None  # not starting soon
    assert rule.after_meal(moves_of(("sleep", at(23, 30), 480), ("study", at(8, 30, 14), 60), ("eat", at(12, 0, 14), 30)), NOW, TZ) is None


def test_the_offer_is_read_back_from_what_the_engine_placed():
    def block(kind, start, minutes, fixed=False):
        return PlanMove(id=f"{kind}{start:%H%M}", kind=kind, title=kind, start=start, end=start + timedelta(minutes=minutes), minutes=minutes, fixed=fixed)

    assert rule.moved_after([block("eat", at(18, 30), 30), block("study", at(19, 1), 60), block("study", at(20, 1), 45)], NOW, TZ) == "dinner"
    assert rule.moved_after([block("study", at(16, 0), 60), block("eat", at(18, 30), 30), block("study", at(19, 1), 60)], NOW, TZ) is None
    assert rule.moved_after([block("eat", at(12, 0), 30), block("study", at(16, 0), 60)], NOW, TZ) is None  # lunch is over: not "after" it
    assert rule.moved_after([block("eat", at(18, 30), 30)], NOW, TZ) is None
    assert rule.move_text("dinner", 1, "study") == MOVE and rule.move_text("lunch", 2, "shift") == "Move your shift blocks to after lunch"


# ---------- the service, on a bus ----------

ITEMS = [Item(connector="canvas", kind="assignment", external_id="a1", title="Project - Milestone1 [COP4533]", when=at(23, 59, 14),
              summary="Due. 100 points."),
         Item(connector="calendar", kind="calendar_event", external_id="c1", title="COP4533 lecture", when=at(10, 40),
              summary="until 11:55 AM · Classroom Building I")]


def night_item(day: date, asleep: int, hr: int, ts: datetime) -> ItemSynced:
    return ItemSynced(event_id=f"night-{day}", ts=ts, connector="health", kind="health", external_id=f"sleep:{day}",
                      title=f"Sleep {asleep // 60} h {asleep % 60} min", when=ts,
                      summary=f"asleep {asleep // 60} h {asleep % 60} min · in bed {asleep // 60} h {asleep % 60 + 20} min · resting HR {hr} bpm")


def synced(ts: datetime, asleep: int = 300, hr: int = 62) -> list[ItemSynced]:
    """The watch's last nine nights, as one sync: ordinary ones, then a short night with the resting heart rate up."""
    return [night_item(TODAY - timedelta(days=i), 450 + (i % 3) * 10, 55 + i % 2, ts) for i in range(8, 0, -1)] + [night_item(TODAY, asleep, hr, ts)]


def expressions(end: datetime, share: float = 0.6, member: str | None = "m-ana", camera: str = "cam-desk") -> list[ExpressionObserved]:
    return [ExpressionObserved(event_id=f"expr-{camera}-{member}-{m}", ts=end - timedelta(minutes=m), camera_id=camera, member_id=member,
                               window_start=end - timedelta(minutes=m + 3), window_end=end - timedelta(minutes=m), dominant="sad",
                               shares={"sad": share, "neutral": 1 - share}, samples=30, confidence=0.8) for m in range(18, -1, -3)]


def seen(ts: datetime, y: float) -> ObservationRecorded:
    ms = int(ts.timestamp() * 1000)
    return ObservationRecorded(event_id=f"obs-{ms}", ts=ts, camera_id="cam-desk", tracks=[person(), bottle(y)], frame_ref=f"frames:cam-desk:{ms}")


def afternoon() -> list:
    """Sat down at 2, picked the bottle up at 2:05, then nothing; looked strained from twenty past three."""
    s = timedelta(seconds=1)
    events = [zone(at(14, 0), "desk"), *(seen(at(14, 0) + s * k, 0.70) for k in (1, 2, 3)), seen(at(14, 5), 0.50), seen(at(14, 5) + 2 * s, 0.50),
              seen(at(14, 6), 0.70), *synced(at(14, 10)), seen(at(15, 30), 0.70), *expressions(at(15, 39)),
              zone(at(15, 39, ) + 30 * s, "desk", "dwell")]
    return sorted(events, key=lambda e: e.ts)


class Ntfy:
    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []

    async def publish(self, topic, message, title=None, tags=None) -> None:
        self.sent.append((title, message))


def box(places, tmp_path, name: str = "a", items=ITEMS) -> tuple[DayLoop, EventBus, list, Ntfy]:  # noqa: F811
    memory = FakeMemory(items=items)
    hub = ConnectorHub([], factory=never)
    dayrun = DayRun(places, memory=memory, habits=None, hub=hub, data_dir=tmp_path / name, clock=lambda: NOW)
    loop = DayLoop(dayrun, places, memory=memory, hub=hub, clock=CLOCK)
    bus = EventBus(tmp_path / "events")
    loop.subscribe(bus)
    ntfy, out = Ntfy(), []
    bus.subscribe("day.", day_phone(ntfy, "atlas", narrated=True))
    bus.subscribe("day.", out.append)
    return loop, bus, out, ntfy


async def lived(places, tmp_path, events=None, name: str = "a") -> tuple[DayLoop, EventBus, list, Ntfy]:  # noqa: F811
    """The afternoon as it is in the log by 3:40 PM, then the first look at the rule: so every signal is in."""
    loop, bus, out, ntfy = box(places, tmp_path, name)
    await bus.publish_all(loop.planner.plan(at(8, 10)))
    loop.checkin.enabled = False
    for e in afternoon() if events is None else events:
        await bus.publish(e)
    loop.checkin.enabled = True
    await bus.publish_all(loop.tick(NOW))
    return loop, bus, out, ntfy


def of(out: list, kind) -> list:
    return [e for e in out if isinstance(e, kind)]


def test_an_afternoon_with_agreeing_signals_raises_one_check_in_in_words(places, tmp_path):  # noqa: F811
    loop, bus, out, ntfy = asyncio.run(lived(places, tmp_path))
    [e] = of(out, DayCheckinRaised)
    assert e.type in BOX_ONLY_EVENTS and e.day == TODAY and e.ts == NOW == e.at
    assert [s.kind for s in e.signals] == ["sleep", "resting_hr", "expression", "sitting", "drink"] and e.history is None
    assert [s.summary for s in e.signals] == ["Sleep: short", "Heart rate: up", "Looked strained 20 min", "At the desk 100 min", "No drink since 2:05 PM"]
    assert e.signals[0].evidence_refs == [f"health:sleep:{TODAY}"] and e.signals[2].evidence_refs[0].startswith("expr-cam-desk")
    assert e.signals[4].since == at(14, 5) + timedelta(seconds=2) and e.signals[4].evidence_refs[0].startswith("frames:cam-desk:")
    assert [s.kind for s in e.suggestions] == ["drink", "walk", "move_block"] and e.proposes_replan
    assert e.suggestions[2].text == "Move the study block to after dinner"
    assert e.text.startswith("You slept about 5 h, your resting heart rate is up, and you've looked strained for the last 20 minutes.")
    assert "62" not in e.model_dump_json() and "bpm" not in e.model_dump_json()  # never the heart rate itself
    [told] = [n for n in of(out, DayNarrated) if n.moment == "checkin"]
    assert told.ref == e.event_id and told.display == e.text and not told.urgent
    assert told.spoken.startswith("You slept about five hours, your resting heart rate is up, and you've looked strained")
    assert told.spoken.endswith("Drink some water, take ten minutes outside, and I can move the study block to after dinner if you say yes.")
    # the plan did not move: nothing was proposed, and the day is measured against the morning's plan
    assert of(out, DayReplanProposed) == [] and loop.planner.pending == {}
    # the push names what to do and never why
    [(title, message)] = [m for m in ntfy.sent if m[0] == "Check-in"]
    assert message == "Atlas has a check-in for you. Drink some water, take ten minutes outside. Open Now to see why." == checkin_push(e)
    assert not [w for w in ("slept", "heart", "strained", "5 h") if w in " ".join(m for _, m in ntfy.sent if "check-in" in m)]


def test_the_cooldown_a_class_and_an_empty_desk_keep_it_quiet(places, tmp_path):  # noqa: F811
    async def run():
        loop, bus, out, _ = await lived(places, tmp_path)
        c = loop.checkin
        assert len(c.raised) == 1
        for minutes in (5, 45, 85):  # still strained, still sitting: nothing more for ninety minutes
            later = NOW + timedelta(minutes=minutes)
            for e in [zone(later, "desk", "dwell"), *expressions(later)]:
                e.event_id += f"-{minutes}"
                await bus.publish(e)
            await bus.publish_all(loop.tick(later + timedelta(minutes=4)))
        assert len(c.raised) == 1
        again = NOW + timedelta(minutes=95)
        for e in [zone(again, "desk", "dwell"), *expressions(again)]:
            e.event_id += "-again"
            await bus.publish(e)
        assert len(c.raised) == 2 and c.raised[1].ts == again
        assert len([n for n in of(out, DayNarrated) if n.moment == "checkin"]) == 2  # said once each, also with the same words
        return loop

    asyncio.run(run())

    async def in_class():
        late = [Item(connector="calendar", kind="calendar_event", external_id="c2", title="Office hours", when=at(15, 0), summary="until 4:30 PM · Classroom Building I")]
        loop, bus, out, _ = box(places, tmp_path / "class", items=ITEMS + late)
        await bus.publish_all(loop.planner.plan(at(8, 10)))
        for e in afternoon():
            await bus.publish(e)
        await bus.publish_all(loop.tick(NOW))
        assert loop.checkin.raised == [] and loop.checkin.watching(NOW)["held"] == "during Office hours"
        assert loop.checkin.watching(NOW)["agree"] is True

    asyncio.run(in_class())

    async def left():
        loop, bus, out, _ = box(places, tmp_path / "left")
        await bus.publish_all(loop.planner.plan(at(8, 10)))
        events = [e for e in afternoon() if e.ts < at(15, 0)] + [zone(at(15, 0), "desk", "exit")] + expressions(at(15, 39))
        for e in events:
            await bus.publish(e)
        await bus.publish_all(loop.tick(NOW))
        assert loop.checkin.raised == [] and loop.checkin.watching(NOW)["held"] == "nobody at the desk"

    asyncio.run(left())


def test_one_signal_or_none_never_raises_on_the_bus(places, tmp_path):  # noqa: F811
    async def run(events) -> DayLoop:
        loop, bus, out, _ = box(places, tmp_path / str(len(events)))
        await bus.publish_all(loop.planner.plan(at(8, 10)))
        for e in events:
            await bus.publish(e)
        await bus.publish_all(loop.tick(NOW))
        return loop

    sitting = [zone(at(14, 0), "desk"), zone(at(15, 39), "desk", "dwell")]
    assert asyncio.run(run(sitting)).checkin.raised == []  # a long sitting alone
    watch_only = [zone(at(15, 30), "desk"), *synced(at(15, 31)), zone(at(15, 39), "desk", "dwell")]
    assert asyncio.run(run(watch_only)).checkin.raised == []  # a short night and a heart rate, nothing happening now
    anonymous = sitting[:1] + expressions(at(15, 38), member=None) + expressions(at(15, 38), camera="cam-door") + sitting[1:]
    assert asyncio.run(run(anonymous)).checkin.looks == {}  # the room's faces and another camera are nobody's


def test_a_restart_replays_the_check_in_and_raises_and_pushes_nothing_again(places, tmp_path):  # noqa: F811
    before, _, out, ntfy = asyncio.run(lived(places, tmp_path))
    assert len(before.checkin.raised) == 1 and len([m for m in ntfy.sent if m[0] == "Check-in"]) == 1

    async def restarted():
        loop, bus, out, ntfy = box(places, tmp_path, name="b")
        await bus.replay_into()
        return loop, bus, out, ntfy

    after, bus, replayed, pushed = asyncio.run(restarted())
    c = after.checkin
    assert c.raised == before.checkin.raised and c.nights == before.checkin.nights and c.looks == before.checkin.looks
    assert c.drinks.moments == before.checkin.drinks.moments and c.drinks.container == before.checkin.drinks.container
    assert replayed == [] and pushed.sent == []  # replay published nothing and pushed nothing
    assert after.narrator.latest["checkin"] == before.narrator.latest["checkin"] and after.narrator.checkin == c.raised[0]
    types = [e.type for e in EventBus(tmp_path / "events").read_log()]
    assert types.count("day.checkin.raised") == 1 and types.count("day.narrated") == sum(isinstance(e, DayNarrated) for e in out)
    # the restarted box is inside the cooldown the log recorded: its first ticks raise nothing
    assert after.tick(NOW + timedelta(minutes=1)) == [] or not of(after.tick(NOW + timedelta(minutes=1)), DayCheckinRaised)
    assert not of(asyncio.run(bus.publish_all(after.tick(NOW + timedelta(minutes=10)))) or [], DayCheckinRaised) and len(c.raised) == 1
    assert c.watching(NOW + timedelta(minutes=10))["held"] == "a check-in was raised less than ninety minutes ago"


def test_someone_who_is_forgotten_or_turns_expressions_off_is_forgotten_here(places, tmp_path):  # noqa: F811
    loop, *_ = box(places, tmp_path)
    for e in expressions(NOW):
        loop.checkin.on_expression(e)
    assert set(loop.checkin.looks) == {"m-ana"}
    loop.checkin.on_identity(IdentityChanged(enabled=True, members=["m-ben"]))
    assert loop.checkin.looks == {}


def test_it_can_be_switched_off(places, tmp_path):  # noqa: F811
    memory = FakeMemory(items=ITEMS)
    hub = ConnectorHub([], factory=never)
    dayrun = DayRun(places, memory=memory, habits=None, hub=hub, data_dir=tmp_path, clock=lambda: NOW)
    loop = DayLoop(dayrun, places, memory=memory, hub=hub, clock=CLOCK, narration={"checkin": False})
    assert loop.checkin.enabled is False and loop.checkin.check(NOW) == []


# ---------- the API ----------


def test_the_api_serves_the_card_its_evidence_and_the_two_buttons(places, tmp_path):  # noqa: F811
    async def run():
        loop, bus, out, _ = await lived(places, tmp_path)
        app = FastAPI()
        app.include_router(day_router(loop, publish=bus.publish, memory=loop.memory, clock=lambda: NOW + timedelta(minutes=2)))
        return loop, out, TestClient(app)

    loop, out, client = asyncio.run(run())
    [e] = of(out, DayCheckinRaised)
    view = client.get("/api/day/checkin").json()
    card = view["latest"]
    assert card["event_id"] == e.event_id and card["answered"] is None and card["display"] == e.text
    assert card["spoken"].startswith("You slept about five hours")
    ev = card["evidence"]
    assert set(ev) == {"sleep", "resting_hr", "expression", "sitting", "drink"}
    assert ev["sleep"]["asleep_min"] == 300 and ev["sleep"]["usual_min"] == 460 and ev["sleep"]["nights"] == 8
    # the heart rate's shape against the person's own baseline, and no value until the phone asks to see numbers
    assert ev["resting_hr"]["series"][-1] == {"day": str(TODAY), "rel": round(62 / 55.5, 3)} and len(ev["resting_hr"]["series"]) == 9
    assert "values" not in ev["resting_hr"] and "today" not in ev["resting_hr"] and "62" not in str(view).replace(str(round(62 / 55.5, 3)), "")
    shown = client.get("/api/day/checkin?numbers=true").json()["latest"]["evidence"]["resting_hr"]
    assert shown["today"] == 62 and shown["baseline"] == 55.5 and len(shown["values"]) == 9
    assert len(ev["expression"]["windows"]) == 7 and ev["expression"]["windows"][0]["strained"] == 0.6
    assert ev["sitting"] == {"since": at(14, 0).isoformat(), "minutes": 100}
    assert ev["drink"]["picked_up"] and ev["drink"]["frame_ref"].startswith("frames:cam-desk:") and ev["drink"]["last_seen"]["label"] == "bottle"
    assert view["watching"]["held"] == "a check-in was raised less than ninety minutes ago" and view["rule"]["cooldown_min"] == 90
    assert view["history"][0]["id"] == e.event_id and any("not a diagnosis" in line or "diagnosis" in line for line in view["limits"])
    assert client.get("/api/day/now").json()["checkin"]["event_id"] == e.event_id

    # "Move the block": a re-plan with its card; the plan itself has not moved
    current = loop.planner.current[TODAY].plan_id
    moved = client.post("/api/day/checkin/move", json={"id": e.event_id}).json()
    replan = loop.planner.pending[TODAY]
    assert moved["answered"] == "move" and moved["plan_id"] == replan.plan_id and moved["action_id"] == replan.action_id
    assert replan.asked and replan.why == "You asked to move the study block to after dinner."
    assert loop.planner.current[TODAY].plan_id == current and replan.plan_id not in loop.planner.committed
    today = [m for m in replan.moves if m.start.astimezone(TZ).date() == TODAY]
    dinner = max(m.end for m in today if m.kind == "eat")
    assert all(m.start >= dinner for m in today if m.kind == "study") and any(m.kind == "study" for m in today)
    assert not [w for w in ("slept", "heart", "strained") if w in replan.why]  # the card's reason says nothing about the person
    assert client.get("/api/day/checkin").json()["latest"]["answered"] == "move"
    assert [a.answer for a in of(out, DayCheckinAnswered)] == ["move"]

    assert client.post("/api/day/checkin/done", json={}).json() == {"id": e.event_id, "answered": "done"}
    assert loop.checkin.answers[e.event_id].answer == "done"
    assert client.post("/api/day/checkin/done", json={"id": "nope"}).status_code == 404
    # hours later it is history, not something to show
    app = FastAPI()
    app.include_router(day_router(loop, clock=lambda: NOW + timedelta(hours=5)))
    later = TestClient(app).get("/api/day/checkin").json()
    assert later["latest"] is None and later["history"][0]["answered"] == "done"


def test_the_api_with_no_check_in_yet(places, tmp_path):  # noqa: F811
    loop, bus, out, _ = box(places, tmp_path)
    app = FastAPI()
    app.include_router(day_router(loop, clock=lambda: NOW))
    client = TestClient(app)
    view = client.get("/api/day/checkin").json()
    assert view["latest"] is None and view["history"] == [] and view["watching"]["signals"] == []
    assert client.post("/api/day/checkin/done").status_code == 404 and client.post("/api/day/checkin/move").status_code == 404


def test_a_drink_caption_and_the_desks_object_list_reach_the_watch(places, tmp_path):  # noqa: F811
    loop, *_ = box(places, tmp_path)
    c = loop.checkin

    def scene(ts, caption, zone_id=None, objects=(), camera="cam-desk"):
        from contracts.types import ObjectHit

        return SceneDescribed(ts=ts, camera_id=camera, caption=caption, zone_id=zone_id, model_version="t", frame_ref=f"frames:{camera}:1",
                              objects=[ObjectHit(label=o, position="on the desk") for o in objects])

    c.on_scene(scene(at(14, 0), "a person sipping from a mug", camera="cam-door"))  # another camera
    c.on_scene(scene(at(14, 1), "objects on the desk", zone_id="desk", objects=["keys", "mug"]))
    assert c.drinks.moments == [] and c.drinks.container[1] == "mug"
    c.on_scene(scene(at(14, 2), "a person sipping from a mug"))
    assert [m[0] for m in c.drinks.moments] == [at(14, 2)]
