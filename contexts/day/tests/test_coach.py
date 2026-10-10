"""The study coach (D2): the session arithmetic (pure), the note when a block ends, the evening review, the next
morning's plan, replay and the API. Every time below is event time."""

import asyncio
from datetime import datetime, timedelta

from fastapi import FastAPI
from fastapi.testclient import TestClient

from bus import EventBus
from contexts.day.api import day_router
from contexts.day.application.loop import DayLoop
from contexts.day.domain.choose import choose, chunks
from contexts.day.domain.coach import (
    DeskPresence,
    Signal,
    Words,
    block_size,
    day_text,
    mentions,
    note_text,
    review_session,
    stretches,
    suggest,
    trigger_of,
)
from contexts.day.tests.test_day import CLOCK, SUNDAY, TZ, FakeHabits, at, items, make, places, zone  # noqa: F401
from contexts.play.dayrun.service import DayRun
from contexts.profiles.loader import load_profile
from contracts import REGISTRY
from contracts.day import DayCoachNoted, DayPlanProposed, DayReviewed, PlanMove, PlanOutcome
from contracts.memory import Episode, Item
from contracts.perception import SceneDescribed
from contracts.types import ObjectHit

DESK = "apartment.desk"
OUTCOME = PlanOutcome(won=True, stars=2, points=80, summary="Win, 2 stars.", sleep_hours=8, meals=2, study_minutes=165,
                      energy_min=40, deadline_risk=0.2)


def block(start: datetime, minutes: int, id: str = "m0", place: str = DESK, name: str = "Desk", kind: str = "study") -> PlanMove:
    return PlanMove(id=id, kind=kind, title=f"Study at {name}", start=start, end=start + timedelta(minutes=minutes),
                    minutes=minutes, place=place, place_name=name)


def span(h1: int, m1: int, h2: int, m2: int) -> tuple[datetime, datetime]:
    return at(h1, m1), at(h2, m2)


# The owner's example: 120 minutes planned, 70 done, two breaks, the long one right after the group chat.
MORNING = block(at(9), 120)
AT_DESK = [span(9, 6, 9, 48), span(10, 19, 10, 40), span(10, 53, 11, 0)]
SIGNALS = [Signal("chat", at(9, 46, ) + timedelta(seconds=40), "groupme:friends:1"), Signal("phone", at(9, 47), "ep-desk-7"),
           Signal("food", at(9, 30), "ep-desk-6"), Signal("door", at(10, 37), "ep-door-2")]


# --- at the desk ---


def test_presence_is_the_union_of_person_stays_and_an_open_stay_ends_when_last_seen():
    p = DeskPresence()
    for ts, track, kind in [(at(9), 1, "enter"), (at(9, 10), 2, "enter"), (at(9, 20), 1, "exit"), (at(9, 30), 2, "dwell"),
                            (at(9, 40), 2, "exit"), (at(10), 3, "enter"), (at(10, 5), 3, "dwell"), (at(10, 30), 9, "exit")]:
        p.add(ts, track, kind)
    assert p.intervals(at(8), at(12)) == [span(9, 0, 9, 40), span(10, 0, 10, 5)]  # two people count once; track 3 never exited
    assert p.intervals(at(9, 15), at(9, 30)) == [span(9, 15, 9, 30)]  # clipped to the block
    p.add(at(11), 4, "dwell", dwell_s=120)  # a dwell whose enter this log does not have
    assert p.intervals(at(10, 30), at(12)) == [span(10, 58, 11, 0)]
    p.prune(at(9, 50))
    assert p.intervals(at(8), at(9, 50)) == []


def test_an_absence_under_a_minute_is_not_a_break():
    spans = [span(9, 0, 9, 20), (at(9, 20) + timedelta(seconds=59), at(9, 40)), (at(9, 41), at(9, 50))]
    assert stretches(spans) == [span(9, 0, 9, 40), span(9, 41, 9, 50)]


# --- one session ---


def test_a_session_counts_desk_minutes_breaks_the_longest_stretch_and_the_late_start():
    s = review_session(MORNING, AT_DESK, SIGNALS, DESK, tz=TZ)
    assert (s.measured, s.planned_minutes, s.desk_minutes) == (True, 120, 70)
    assert (s.late_start_minutes, s.ended_early_minutes) == (6, 0)
    assert (s.longest_stretch_minutes, s.longest_stretch_start) == (42, at(9, 6))
    assert [(b.start, b.end, b.minutes, b.trigger) for b in s.breaks] == [(at(9, 48), at(10, 19), 31, "chat"), (at(10, 40), at(10, 53), 13, "none")]
    assert [(e.kind, e.id) for e in s.breaks[0].evidence] == [("chat", "groupme:friends:1"), ("phone", "ep-desk-7")]
    assert s.text == ("Study at Desk, 9:00 AM to 11:00 AM: you planned 120 minutes and were at the desk for 70. "
                      "You sat down 6 minutes after the block started. "
                      "At 9:48 AM you were away for 31 minutes, right after a message arrived in the group chat (and the phone was in view). "
                      "At 10:40 AM you were away for 13 minutes, with no visible reason. "
                      "What worked: your longest unbroken stretch was 42 minutes, from 9:06 AM.")
    nxt = block(at(13), 45, id="m2")
    assert note_text(s, nxt) == "You did 70 of 120 minutes. Two breaks; the long one came right after the group chat. Next study block is 45 minutes."


def test_the_first_stretch_is_named_as_the_first_and_a_whole_block_is_said_to_have_worked():
    s = review_session(MORNING, [span(9, 0, 9, 42), span(9, 50, 10, 30)], [], DESK, tz=TZ)
    assert "What worked: your longest unbroken stretch was the first 42 minutes." in s.text
    assert "You left the desk 30 minutes before the end." in s.text
    assert note_text(s) == "You did 82 of 120 minutes. One break, for 8 minutes."
    whole = review_session(MORNING, [span(8, 50, 11, 5)], SIGNALS, DESK, tz=TZ)
    assert (whole.desk_minutes, whole.breaks, whole.late_start_minutes) == (120, [], 0)
    assert whole.text.endswith("What worked: you stayed for the whole block.") and note_text(whole) == "You did 120 of 120 minutes. No breaks."


def test_the_trigger_is_what_happened_in_the_two_minutes_before_in_a_fixed_order():
    left = at(10)
    sig = [Signal("food", left - timedelta(seconds=30), "ep-1"), Signal("door", left - timedelta(seconds=100), "ep-2"),
           Signal("door", left - timedelta(seconds=90), "ep-2"),  # the same episode twice is one piece of evidence
           Signal("chat", left - timedelta(seconds=121), "groupme:x:1"),  # too early
           Signal("phone", left + timedelta(seconds=5), "ep-3")]  # after the exit: not a reason for it
    trigger, evidence = trigger_of(left, sig)
    assert trigger == "door" and [(e.kind, e.id) for e in evidence] == [("door", "ep-2"), ("food", "ep-1")]
    assert trigger_of(left, sig[3:]) == ("none", [])
    assert trigger_of(left, [*sig, Signal("chat", left - timedelta(seconds=120), "groupme:x:2")])[0] == "chat"
    assert mentions("picks up their phone", ("phone",)) and not mentions("wearing headphones", ("phone",))


def test_a_block_away_from_the_box_is_not_measured_and_its_numbers_are_none_not_zero():
    library = block(at(14), 180, place="ucf.library", name="John C. Hitt Library")
    away = review_session(library, [span(14, 30, 14, 33)], SIGNALS, DESK, tz=TZ)  # someone passed the desk for three minutes
    assert not away.measured and away.not_measured == "planned at John C. Hitt Library, away from the box"
    assert (away.desk_minutes, away.late_start_minutes, away.longest_stretch_minutes, away.breaks) == (None, None, None, [])
    assert away.text == "Study at John C. Hitt Library, 2:00 PM to 5:00 PM: not measured (planned at John C. Hitt Library, away from the box)."
    stayed = review_session(library, [span(14, 0, 14, 50)], [], DESK, tz=TZ)  # planned there, done at the desk
    assert stayed.measured and stayed.desk_minutes == 50
    assert "It was planned at John C. Hitt Library; the camera saw the desk in use, so it is measured there." in stayed.text
    nobody = review_session(MORNING, [], SIGNALS, DESK, tz=TZ)  # at the box, and nobody came
    assert nobody.measured and nobody.desk_minutes == 0 and "nobody was seen at the desk (or the camera was off)" in nobody.text


# --- the pattern ---


def test_suggestions_are_drawn_from_the_pattern_and_never_more_than_two():
    s = review_session(MORNING, AT_DESK, SIGNALS, DESK, tz=TZ)
    advice = suggest([s])
    assert advice.suggestions == [
        "Hold chat notifications until the break: 1 of your 2 breaks (31 minutes) began right after the group chat.",
        "Plan shorter blocks: your longest unbroken stretch was 42 minutes, so I'm planning 45-minute blocks."]
    assert (advice.block_minutes, advice.longest_stretch) == (45, 42)
    phone = review_session(MORNING, [span(9, 15, 10, 0), span(10, 30, 11, 0)], [Signal("phone", at(9, 59), "ep-9")], DESK, tz=TZ)
    assert suggest([phone]).suggestions[0].startswith("Keep the phone face down or outside the desk zone during a block: 1 of your 1 break (30 minutes)")
    late = review_session(MORNING, [span(9, 20, 11, 0)], [], DESK, tz=TZ)
    assert suggest([late]).suggestions == ["Start with the hardest thing the moment the block opens: the first sit-down came 20 minutes in."]
    whole = review_session(MORNING, [span(9, 0, 11, 0)], [], DESK, tz=TZ)
    assert suggest([whole]) == suggest([whole]) and suggest([whole]).suggestions == ["Keep this shape: nothing pulled you away for long."]
    assert suggest([whole]).block_minutes is None
    away = review_session(block(at(14), 60, place="ucf.library", name="Library"), [], [], DESK, tz=TZ)
    assert suggest([away]).suggestions == []  # nothing measured: nothing to say
    assert (block_size(42), block_size(10), block_size(200), block_size(45)) == (45, 25, 120, 45)


def test_the_words_come_from_the_profile_not_from_the_code(places):  # noqa: F811
    words = Words(activity="work", workplace="station")
    s = review_session(MORNING.model_copy(update={"title": "Work at Station"}), AT_DESK, SIGNALS, DESK, words, TZ)
    assert "were at the station for 70" in s.text and "desk" not in s.text
    assert note_text(s, block(at(13), 45), words).endswith("Next work block is 45 minutes.")
    assert "outside the station zone" in suggest([review_session(MORNING, [span(9, 15, 10, 0), span(10, 30, 11, 0)],
                                                               [Signal("phone", at(9, 59), "e")], DESK, words, TZ)], words).suggestions[0]
    text = day_text([s], words, habits_desk_minutes=95, tz=TZ)
    assert text.startswith("You planned 120 minutes of work in 1 block the camera could see, and were at the station for 70.")
    assert text.endswith("Habits counted 95 minutes at the station over the whole day.")
    shift = load_profile("config", "shift-worker")
    loop = DayLoop(DayRun(places, profile=shift), places, clock=CLOCK)
    assert (loop.coach.words.activity, loop.coach.words.workplace) == (shift.vocabulary.activity, shift.vocabulary.workplace)
    assert loop.coach.words != Words()  # not the student's


# --- the coach in the loop ---


def plan_with(*blocks: PlanMove, day=SUNDAY, made: datetime | None = None) -> DayPlanProposed:
    made = made or at(8, 10)
    return DayPlanProposed(event_id=f"day-plan-test-{day}", ts=made, day=day, plan_id=f"plan-{day}-0810", start=made,
                           end=made + timedelta(hours=22), moves=list(blocks), outcome=OUTCOME, text="A test day.")


def desk(minute_spans: list[tuple[datetime, datetime]], track: int = 1) -> list:
    out = []
    for n, (a, b) in enumerate(minute_spans):
        out += [zone(a, "desk", "enter"), zone(b - timedelta(seconds=3), "desk", "dwell"), zone(b, "desk", "exit")]
        for e in out[-3:]:
            e.track_id = track + n
    return out


def scene(ts: datetime, caption: str, objects: list[tuple[str, str]] = (), zone_id: str | None = None) -> SceneDescribed:
    return SceneDescribed(ts=ts, camera_id="cam-desk", caption=caption, zone_id=zone_id, frame_ref=f"frames:cam-desk:{int(ts.timestamp() * 1000)}",
                          model_version="test", objects=[ObjectHit(label=label, position=pos) for label, pos in objects])


CHAT = Item(connector="groupme", kind="message", external_id="friends:9", title="Riley: are you coming out tonight?",
            when=at(9, 46) + timedelta(seconds=40))
EPISODES = [Episode(id="ep-desk-7", camera_id="cam-desk", start=at(9, 44), end=at(9, 48),
                    captions=["A person types, wearing headphones.", "The person picks up their phone."],
                    keyframe_refs=[f"frames:cam-desk:{int(at(9, 44).timestamp() * 1000)}", f"frames:cam-desk:{int(at(9, 47).timestamp() * 1000)}"])]
DAY = [*desk(AT_DESK), scene(at(9, 47), "The person picks up their phone."), scene(at(9, 30), "A person types on a laptop."),
       scene(at(10, 39), "desk: keys, phone", [("keys", "left"), ("phone", "face up")], zone_id="desk")]


def coached(items, places, tmp_path, habits=None):  # noqa: F811
    loop = make([*items, CHAT], places, tmp_path, habits=habits or FakeHabits(focus=1.2), episodes=EPISODES)
    loop.planner.on_plan_proposed(plan_with(MORNING, block(at(13), 45, id="m2"), block(at(14), 180, id="lib", place="ucf.library", name="Library")))
    for e in sorted(DAY, key=lambda e: e.ts):
        (loop.coach.on_scene if isinstance(e, SceneDescribed) else loop.coach.on_zone)(e)
    return loop


def test_a_note_goes_out_once_when_a_measured_block_ends(items, places, tmp_path):  # noqa: F811
    loop = coached(items, places, tmp_path)
    c = loop.coach
    assert c.check(at(10, 59)) == []  # the block is still on
    (note,) = c.check(at(11, 1))
    assert isinstance(note, DayCoachNoted) and (note.day, note.block_id, note.ts) == (SUNDAY, "m0", at(11, 1))
    assert note.text == "You did 70 of 120 minutes. Two breaks; the long one came right after the group chat. Next study block is 45 minutes."
    assert note.next_block.id == "m2" and note.session.desk_minutes == 70
    first = note.session.breaks[0]
    assert [(e.kind, e.id) for e in first.evidence] == [("chat", "groupme:friends:9"), ("phone", "ep-desk-7")]  # an item id, an episode id
    assert note.session.breaks[1].trigger == "none"  # the phone in the 10:39 list is the list's first sight of it: not "appearing"
    assert note.calibration.ratio == 1.7 and note.calibration.blocks == 1
    assert note.calibration.text == "Your plans run 1.7x what happens (calibration 58/100 over 1 measured block)."
    assert c.check(at(11, 2)) == []  # handed to the bus: not twice
    c.on_noted(note)
    assert c.check(at(11, 3)) == [] and c.latest is note
    # the block away from the box ends without a note, and a block that ended while the box was off is not announced late
    assert c.check(at(17, 1)) == []
    late = coached(items, places, tmp_path / "late")
    assert late.coach.check(at(15)) == []  # m0 ended four hours ago; m2 was skipped, and a skipped block is not announced
    assert [s.block_id for s in late.coach.sessions(SUNDAY, at(15))] == ["m0", "m2"]  # but it is in the day's sessions


def test_a_phone_counts_when_it_appears_in_the_desk_list_not_while_it_lies_there(items, places, tmp_path):  # noqa: F811
    loop = make(items, places, tmp_path)
    c = loop.coach
    for ts, labels in [(at(9, 10), ["keys"]), (at(9, 20), ["keys", "phone"]), (at(9, 30), ["keys", "phone"]), (at(9, 40), ["keys", "can"])]:
        c.on_scene(scene(ts, "desk: things", [(label, "on the desk") for label in labels], zone_id="desk"))
    c.on_scene(scene(at(9, 50), "A person opens an energy drink.", ))
    c.on_scene(SceneDescribed(ts=at(9, 51), camera_id="cam-door", caption="A person on the phone walks past.", frame_ref="frames:cam-door:1", model_version="t"))
    assert [(kind, ts) for kind, ts, _ in c.seen] == [("phone", at(9, 20)), ("food", at(9, 40)), ("food", at(9, 50))]


def test_the_evening_review_carries_the_sessions_and_the_next_plan_reads_the_advice_back(items, places, tmp_path):  # noqa: F811
    loop = coached(items, places, tmp_path)
    p = loop.planner
    (note,) = loop.coach.check(at(11, 1))
    loop.coach.on_noted(note)
    review = next(e for e in p.tick(at(20)) if isinstance(e, DayReviewed))
    assert [(s.block_id, s.measured, s.desk_minutes) for s in review.sessions] == [("m0", True, 70), ("m2", True, 0), ("lib", False, None)]
    assert review.coach_text.startswith("You planned 165 minutes of study in 2 blocks the camera could see, and were at the desk for 70.")
    assert "31 minutes away from 9:48 AM, right after a message arrived in the group chat (and the phone was in view)." in review.coach_text
    assert "Not measured: 1 block at Library (away from the box)." in review.coach_text
    assert "Habits counted 72 minutes at the desk over the whole day." in review.coach_text
    assert review.coach_text.endswith("Your plans run 2.4x what happens (calibration 29/100 over 2 measured blocks); I'm planning 45-minute blocks.")
    assert review.suggestions == note.suggestions and review.block_minutes == 45 and review.calibration.blocks == 2
    p.on_reviewed(review)

    monday = p.plan(at(8, 5, day=12))[0]
    (source,) = [s for s in monday.sources if s.kind == "review" and s.origin == "coach"]
    assert source.id == f"coach:{review.plan_id}" and "so I'm planning 45-minute blocks" in source.text and "Hold chat notifications" in source.text
    assert "Your plans run 2.4x what happens." in source.text and "Coach: " + source.text.removeprefix("coach: ") in monday.text
    study = [m.minutes for m in monday.moves if m.kind == "study" and not m.fixed]
    assert study and max(study) < 45 + 15 and study.count(45) >= 2  # blocks sized to the longest stretch
    plain = make(items, places, tmp_path / "plain", habits=FakeHabits(focus=1.2)).planner.plan(at(8, 5, day=12))[0]
    assert max(m.minutes for m in plain.moves if m.kind == "study" and not m.fixed) > 60  # without the coach's advice
    assert monday.outcome.study_minutes == plain.outcome.study_minutes  # the same minutes, cut differently


def test_blocks_are_cut_to_the_coachs_size_and_a_short_remainder_joins_the_last_one(items, places, tmp_path):  # noqa: F811
    assert chunks(245, 25, 15) == [25] * 9 + [20] and chunks(235, 25, 15) == [25] * 8 + [35] and chunks(80, 25, 15) == [25, 25, 30] and chunks(100, 45, 15) == [45, 55]
    assert chunks(110, 45, 15) == [45, 45, 20] and chunks(20, 45, 15) == [20]
    loop = make(items, places, tmp_path, habits=FakeHabits())
    board = loop.planner.dayrun.board(at(8, 10))
    whole, a = choose(loop.planner.dayrun.cfg, loop.planner.dayrun.walks, board)
    cut, b = choose(loop.planner.dayrun.cfg, loop.planner.dayrun.walks, board, study_chunk=45)
    assert max(m.minutes or 0 for m in cut if m.kind == "study") < 60 < max(m.minutes or 999 for m in whole if m.kind == "study")
    assert (b.outcome.won, b.totals.study_minutes) == (a.outcome.won, a.totals.study_minutes) and not b.rejected


def test_a_restart_rebuilds_the_coach_from_the_log_and_notes_nothing_twice(items, places, tmp_path):  # noqa: F811
    async def live() -> DayLoop:
        bus = EventBus(tmp_path / "events")
        loop = make([*items, CHAT], places, tmp_path / "a", habits=FakeHabits(focus=1.2), episodes=EPISODES)
        loop.subscribe(bus)
        await bus.publish(plan_with(MORNING, block(at(13), 45, id="m2")))
        for e in sorted(DAY, key=lambda e: e.ts):
            await bus.publish(e)
        await bus.publish(zone(at(11, 2), "door", "enter", camera="cam-door"))  # more life after the block: nothing more to say
        return loop

    async def restarted() -> DayLoop:
        bus = EventBus(tmp_path / "events")
        loop = make([*items, CHAT], places, tmp_path / "b", habits=FakeHabits(focus=1.2), episodes=EPISODES)
        loop.subscribe(bus)
        await bus.replay_into()
        return loop

    before, after = asyncio.run(live()), asyncio.run(restarted())
    notes = [e for e in EventBus(tmp_path / "events").read_log() if e.type == "day.coach.noted"]
    assert len(notes) == 1 and notes[0].ts == at(11) and notes[0].session.desk_minutes == 70  # at the event that ended the block
    assert after.coach.notes == before.coach.notes and after.coach.latest == notes[0]
    assert after.coach.seen == before.coach.seen
    assert after.coach.sessions(SUNDAY, at(11, 2)) == before.coach.sessions(SUNDAY, at(11, 2)) == [notes[0].session]
    assert after.coach.tick(at(11, 3)) == []  # replay published nothing and the restarted box does not say it again


def test_coach_events_carry_ids_and_minutes_never_a_picture_or_a_message(items, places, tmp_path):  # noqa: F811
    loop = coached(items, places, tmp_path)
    (note,) = loop.coach.check(at(11, 1))
    loop.coach.on_noted(note)
    review = next(e for e in loop.planner.tick(at(20)) if isinstance(e, DayReviewed))
    for e in (note, review):
        cls = REGISTRY[e.type]
        assert type(e) is cls and set(e.model_dump()) == set(cls.model_fields)
        text = e.model_dump_json().lower()
        assert "frame" not in text and "riley" not in text and "picks up" not in text


# --- the API ---


def test_the_coach_api_cites_each_piece_of_evidence_with_its_words_and_a_picture(items, places, tmp_path):  # noqa: F811
    loop = coached(items, places, tmp_path)
    app = FastAPI()
    app.include_router(day_router(loop, memory=loop.memory, clock=lambda: at(11, 30)))
    c = TestClient(app)
    assert (c.get("/api/day/now").json()["coach"], c.get("/api/day/now").json()["coach_earlier"]) == (None, [])  # nothing said yet
    (note,) = loop.coach.check(at(11, 1))
    loop.coach.on_noted(note)

    body = c.get("/api/day/coach", params={"day": "2026-10-11"}).json()
    assert body["day"] == "2026-10-11" and body["words"] == {"activity": "study", "workplace": "desk"}
    assert [s["block_id"] for s in body["sessions"]] == ["m0"] and body["sessions"][0]["desk_minutes"] == 70
    chat, phone = body["sessions"][0]["breaks"][0]["evidence"]
    assert chat == {"kind": "chat", "id": "groupme:friends:9", "at": CHAT.when.isoformat(), "label": "a message in the group chat",
                    "title": "Riley: are you coming out tonight?", "frame_ref": None}
    assert phone["title"] == "The person picks up their phone." and phone["frame_ref"] == EPISODES[0].keyframe_refs[1]  # the nearest keyframe
    assert body["text"].startswith("You planned 120 minutes of study in 1 block") and len(body["suggestions"]) == 2
    assert body["calibration"]["ratio"] == 1.7 and body["block_minutes"] == 45 and body["desk_minutes_today"] == 72
    assert [n["text"] for n in body["notes"]] == [note.text] and body["latest"]["block_id"] == "m0"
    assert any("not the screen" in line for line in body["limits"])
    assert c.get("/api/day/coach").json()["sessions"] == body["sessions"]  # today by default

    now = c.get("/api/day/now").json()
    assert now["coach"]["text"] == note.text and now["coach"]["session"]["breaks"][0]["evidence"][1]["frame_ref"]
    assert now["coach_earlier"] == []
    empty = c.get("/api/day/coach", params={"day": "2026-10-05"}).json()
    assert (empty["sessions"], empty["text"], empty["suggestions"], empty["calibration"]) == ([], "", [], None)
