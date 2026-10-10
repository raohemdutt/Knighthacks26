"""The study coach's optional input: how an enrolled member looked across a block (`perception.expression.observed`).
With no such events, or too few, the coach says exactly what it said before."""

from datetime import datetime, timedelta

from contexts.day.domain.coach import EXPRESSION_MIN_SAMPLES, Looks, expression_of, note_text, review_session
from contexts.day.tests.test_coach import AT_DESK, DESK, MORNING, SIGNALS, block, coached
from contexts.day.tests.test_day import TZ, at, items, places  # noqa: F401
from contracts.perception import ExpressionObserved, IdentityChanged


def looks(start: datetime, minutes: int, strained_share: float, every_min: int = 3, samples: int = 30) -> list[Looks]:
    """One stretch of `samples` counted looks every `every_min` minutes, `strained_share` of them strained."""
    return [Looks(start + timedelta(minutes=m), start + timedelta(minutes=m + every_min), samples, strained_share * samples)
            for m in range(0, minutes, every_min)]


def observed(start: datetime, minutes: int, shares: dict[str, float], member: str | None = "m-ana", camera: str = "cam-desk"):
    return [ExpressionObserved(ts=start + timedelta(minutes=m + 3), camera_id=camera, member_id=member, window_start=start + timedelta(minutes=m),
                               window_end=start + timedelta(minutes=m + 3), dominant=max(shares, key=shares.get), shares=shares,
                               samples=30, confidence=0.8) for m in range(0, minutes, 3)]


def test_a_rise_in_strained_looks_is_named_with_the_minute_it_began():
    seen = looks(at(9), 36, 0.1) + looks(at(9, 36), 84, 0.6)
    x = expression_of(MORNING, seen)
    assert (x.rose_after_minutes, x.before_share, x.after_share) == (35, 0.1, 0.6) and x.samples == sum(s.samples for s in seen)
    assert x.text == "You looked more strained after the 35-minute mark; a break around then next time."
    assert 0.4 < x.strained_share < 0.5


def test_too_few_looks_or_a_steady_face_say_nothing_and_a_strained_block_is_said_plainly():
    assert expression_of(MORNING, []) is None
    assert expression_of(MORNING, looks(at(9), 3, 0.9, samples=EXPRESSION_MIN_SAMPLES - 1)) is None  # too thin to say
    assert expression_of(MORNING, looks(at(7), 60, 0.9)) is None  # none of it inside the block
    steady = expression_of(MORNING, looks(at(9), 120, 0.2))
    assert (steady.text, steady.rose_after_minutes, steady.strained_share) == ("", None, 0.2)
    easing = expression_of(MORNING, looks(at(9), 60, 0.6) + looks(at(10), 60, 0.1))  # it got better: no "after the mark"
    assert easing.rose_after_minutes is None and easing.text == ""
    mostly = expression_of(MORNING, looks(at(9), 120, 0.7))
    assert mostly.rose_after_minutes is None and mostly.text.startswith("You looked strained for much of this block")
    wobble = expression_of(MORNING, looks(at(9), 60, 0.2) + looks(at(10), 60, 0.35))  # a small rise is noise
    assert wobble.rose_after_minutes is None and wobble.text == ""
    short = expression_of(block(at(9), 15), looks(at(9), 15, 0.1, samples=40)[:2] + looks(at(9, 6), 9, 0.9, samples=40))
    assert short.rose_after_minutes is None  # under two full buckets on a side: no mark is claimed


def test_the_session_and_the_note_are_unchanged_without_looks_and_gain_one_sentence_with_them():
    plain = review_session(MORNING, AT_DESK, SIGNALS, DESK, tz=TZ)
    assert plain.expression is None
    assert review_session(MORNING, AT_DESK, SIGNALS, DESK, tz=TZ, looks=[]) == plain
    assert review_session(MORNING, AT_DESK, SIGNALS, DESK, tz=TZ, looks=looks(at(9), 3, 0.9, samples=5)) == plain  # too thin

    seen = looks(at(9), 36, 0.1) + looks(at(9, 36), 84, 0.6)
    s = review_session(MORNING, AT_DESK, SIGNALS, DESK, tz=TZ, looks=seen)
    sentence = "You looked more strained after the 35-minute mark; a break around then next time."
    assert s.text == f"{plain.text} {sentence}" and s.model_copy(update={"expression": None, "text": plain.text}) == plain
    nxt = block(at(13), 45, id="m2")
    assert note_text(s, nxt) == ("You did 70 of 120 minutes. Two breaks; the long one came right after the group chat. "
                                 f"{sentence} Next study block is 45 minutes.")
    assert note_text(plain, nxt) == "You did 70 of 120 minutes. Two breaks; the long one came right after the group chat. Next study block is 45 minutes."


def test_the_coach_reads_the_desk_cameras_events_for_an_enrolled_member_and_forgets_a_forgotten_one(items, places, tmp_path):  # noqa: F811
    loop = coached(items, places, tmp_path)
    c = loop.coach
    before = c.session(MORNING)
    limits = c.limits()
    assert before.expression is None

    for e in (observed(at(9), 120, {"happy": 1.0}, member=None)  # the anonymous room: nobody's
              + observed(at(9), 120, {"sad": 1.0}, camera="cam-door")):  # another camera: not the desk
        c.on_expression(e)
    assert c.looks == {} and c.session(MORNING) == before and c.limits() == limits

    for e in observed(at(9), 36, {"neutral": 0.9, "sad": 0.1}) + observed(at(9, 36), 84, {"neutral": 0.4, "angry": 0.3, "sad": 0.3}):
        c.on_expression(e)
    for e in observed(at(9), 9, {"happy": 1.0}, member="m-ben"):  # someone seen far less in this block
        c.on_expression(e)
    s = c.session(MORNING)
    assert s.expression.rose_after_minutes == 35 and s.expression.samples == 1200 and "looked more strained after the 35-minute mark" in s.text
    assert s.model_copy(update={"expression": None, "text": before.text}) == before  # nothing else moved
    (note,) = c.check(at(11, 1))
    assert "You looked more strained after the 35-minute mark" in note.text and note.session.expression.before_share == 0.1
    assert len(c.limits()) == len(limits) + 1 and "not an emotion" in c.limits()[-1]

    c.on_identity(IdentityChanged(ts=at(12), enabled=True, members=["m-ben"]))  # "forget me"
    assert list(c.looks) == ["m-ben"]
    assert c.session(MORNING).expression.text == "" and c.session(MORNING).text == before.text  # nine calm minutes: no sentence
    c.on_identity(IdentityChanged(ts=at(12), enabled=True, members=[]))
    assert c.looks == {} and c.session(MORNING) == before and c.limits() == limits


def test_the_loop_subscribes_the_coach_to_expression_and_identity_events(items, places, tmp_path):  # noqa: F811
    import asyncio

    from bus import EventBus

    loop = coached(items, places, tmp_path)
    bus = EventBus(tmp_path / "events-x")
    loop.subscribe(bus)
    asyncio.run(bus.publish_all(observed(at(9), 120, {"neutral": 1.0})))
    assert sum(x.samples for x in loop.coach.looks["m-ana"]) == 1200
    asyncio.run(bus.publish(IdentityChanged(ts=at(12), enabled=False, members=[])))
    assert loop.coach.looks == {}
