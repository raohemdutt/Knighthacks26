"""Optional events ("Near you"), the pure part: reading one from its calendar item, whether it fits the plan (a fixed
event never moves), the state, when Atlas may suggest it, what a yes would move, and the two new narration moments
("suggest", "start") with their grounding check. The read model, the API and the tick that use these are not built
yet: docs/handoff/optional-events-and-reminders.md."""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from contexts.day.domain import optional as o
from contexts.day.domain.narration import BOUNDS, MOMENTS, Words, check, render
from contexts.demo.nearby import PREFIX, nearby_events
from contexts.play.domain.places import PlaceSet
from contexts.play.dayrun.routes import Walks
from contracts import REGISTRY
from contracts.day import BOX_ONLY_EVENTS, DayNarrated, DayOptionalAnswered, DayReplanProposed, PlanMove
from contracts.memory import Item

TZ = ZoneInfo("America/New_York")
DINNER = "nearby:friends-dinner:2026-10-12T17:15:00-04:00"


def at(hour: int, minute: int = 0, day: int = 12) -> datetime:
    return datetime(2026, 10, day, hour, minute, tzinfo=TZ)


@pytest.fixture(scope="module")
def walks() -> Walks:
    return Walks(PlaceSet.load("config/places"))


def item(external_id: str = DINNER, title: str = "Friends' dinner", when: datetime | None = None,
         summary: str = "until 6:15 PM · '63 South · with Riley, Sam, Alex · optional · via GroupMe") -> Item:
    return Item(connector="calendar", kind="calendar_event", external_id=external_id, title=title, when=when or at(17, 15), summary=summary)


def dinner(walks: Walks) -> o.Offer:
    return o.parse(item(), walks.locate, walks.name, TZ)


def move(id: str, kind: str, start: datetime, minutes: int, place: str | None = "ucf.library", fixed: bool = False, title: str | None = None) -> PlanMove:
    return PlanMove(id=id, kind=kind, title=title or f"{kind} block", start=start, end=start + timedelta(minutes=minutes), minutes=minutes,
                    place=place, place_name=place, fixed=fixed)


def walk_of(walks: Walks):
    return lambda a, b: walks.walk(a, b)[0]


# --- reading one ---


def test_an_offer_is_read_from_the_calendar_item(walks):
    d = dinner(walks)
    assert (d.id, d.slug, d.title) == (DINNER, "friends-dinner", "Friends' dinner")
    assert (d.start, d.end, d.minutes) == (at(17, 15), at(18, 15), 60)
    assert (d.place, d.place_name) == ("ucf.dining_63", "'63 South")
    assert d.going == ("Riley", "Sam", "Alex") and d.host is None and d.source == "GroupMe" and d.people == d.going
    assert d.tags == ("with Riley, Sam, Alex", "optional", "via GroupMe")


def test_a_host_comes_first_and_a_place_the_pack_does_not_know_keeps_its_name(walks):
    review = o.parse(item("nearby:cop4533-review:2026-10-12T18:30:00-04:00", "COP4533 review session (optional)", at(18, 30),
                          "until 8:00 PM · John C. Hitt Library · hosted by Andria · with Maya, Priya · optional · via GroupMe"), walks.locate, walks.name, TZ)
    assert review.host == "Andria" and review.people == ("Andria", "Maya", "Priya") and review.place == "ucf.library"
    odd = o.parse(item(summary="until 6:15 PM · Somebody's porch · free"), walks.locate, walks.name, TZ)
    assert odd.place is None and odd.place_name == "Somebody's porch" and odd.going == () and odd.tags == ("free",)
    bare = o.parse(item(summary=None), walks.locate, walks.name, TZ)
    assert bare.end == at(18, 15) and bare.place_name is None  # an hour when the item gives no end


def test_only_a_nearby_calendar_item_with_a_start_is_an_offer(walks):
    assert o.parse(item("outlook:kh-dinner:x"), walks.locate, walks.name, TZ) is None
    assert o.parse(Item(connector="calendar", kind="calendar_event", external_id=DINNER, title="x", when=None), walks.locate, walks.name, TZ) is None


def test_every_seeded_optional_event_parses_with_its_place(walks):
    seeded = [e for e in nearby_events(at(9, 30, day=11)) if e.external_id.startswith(PREFIX)]
    offers = [o.parse(e, walks.locate, walks.name, TZ) for e in seeded]
    assert len(offers) == 7 and all(x is not None and x.place and x.end > x.start for x in offers)
    assert {x.slug for x in offers if x.people} >= {"friends-dinner", "cop4533-review", "board-games"}


def test_the_block_of_an_accepted_event_has_an_id_no_nearby_guard_skips():
    block = o.block_id(DINNER)
    assert block == "optional:friends-dinner:2026-10-12T17:15:00-04:00" and not block.startswith(o.PREFIX)
    assert o.offer_id(block) == DINNER and o.offer_id("m3") is None


# --- fit and displacement ---


def test_nothing_in_the_way_fits_and_the_walk_is_from_where_the_plan_has_you(walks):
    plan = [move("m0", "study", at(15, 0), 60), move("m1", "study", at(18, 30), 25)]
    f = o.fit(dinner(walks), plan, at(14), walk_of(walks), seen_place="apartment.desk")
    assert f.fits == "yes" and f.ok and f.displaces == () and f.conflicts == ()
    assert f.from_place == "ucf.library" and f.walk_minutes == walks.walk("ucf.library", "ucf.dining_63")[0] == 5
    assert f.leave_by == at(17, 10) and f.before.id == "m1"  # "it fits before your study block"


def test_with_nothing_planned_before_it_the_walk_is_from_where_the_cameras_saw_you(walks):
    f = o.fit(dinner(walks), [move("m1", "study", at(18, 30), 25)], at(14), walk_of(walks), seen_place="apartment.desk")
    assert f.from_place == "apartment.desk" and f.walk_minutes == 16 and f.leave_by == at(16, 59)
    nowhere = o.fit(dinner(walks), [], at(14), walk_of(walks))
    assert nowhere.fits == "yes" and nowhere.walk_minutes == 0 and nowhere.leave_by is None and nowhere.before is None


def test_a_suggested_block_in_the_way_is_nearly_and_says_what_would_move(walks):
    plan = [move("m0", "study", at(16, 30), 60, title="Study at John C. Hitt Library"), move("m1", "eat", at(18, 0), 30, place="ucf.union")]
    f = o.fit(dinner(walks), plan, at(14), walk_of(walks))
    assert f.fits == "nearly" and f.ok and [t.id for t in f.displaces] == ["m0", "m1"] and f.conflicts == ()
    assert f.displaces[0].overlap_minutes == 20 and f.displaces[1].overlap_minutes == 15  # from the time to leave, to its end
    assert f.reason == "Study at John C. Hitt Library would move"


def test_a_fixed_event_in_the_way_is_a_no_and_is_never_listed_as_something_to_move(walks):
    plan = [move("outlook:lab", "class", at(17, 0), 90, place="ucf.eng2", fixed=True, title="Lab"), move("m0", "study", at(18, 30), 25)]
    f = o.fit(dinner(walks), plan, at(14), walk_of(walks))
    assert f.fits == "no" and not f.ok and [t.id for t in f.conflicts] == ["outlook:lab"] and "Lab, which is fixed" in f.reason
    assert all(not t.fixed for t in f.displaces)


def test_leaving_a_little_early_for_the_next_fixed_event_is_nearly_and_a_lot_is_a_no(walks):
    back = walks.walk("ucf.dining_63", "apartment.desk")[0]
    assert back == 16
    close = [move("outlook:talk", "class", at(18, 15) + timedelta(minutes=back - 10), 60, place="apartment.desk", fixed=True, title="Talk")]
    f = o.fit(dinner(walks), close, at(14), walk_of(walks))
    assert f.fits == "nearly" and f.leave_early_minutes == 10 and f.conflicts == () and f.reason == "you would leave 10 minutes early for Talk"
    tight = [move("outlook:talk", "class", at(18, 15), 60, place="apartment.desk", fixed=True, title="Talk")]  # 16 minutes early: too much
    assert o.fit(dinner(walks), tight, at(14), walk_of(walks)).fits == "no"


def test_too_late_to_walk_there_is_a_no_and_its_own_block_is_not_in_its_way(walks):
    assert o.fit(dinner(walks), [], at(17, 5), walk_of(walks), seen_place="apartment.desk").fits == "no"
    mine = move(o.block_id(DINNER), "eat", at(17, 15), 60, place="ucf.dining_63", fixed=True, title="Friends' dinner")
    assert o.fit(dinner(walks), [mine], at(14), walk_of(walks)).fits == "yes"


# --- the state, and when Atlas may bring it up ---


def test_the_four_states(walks):
    d = dinner(walks)
    assert o.state_of(d, at(16), None, False) == "suggested"
    assert o.state_of(d, at(16), "yes", False) == "suggested"  # a yes whose card was never approved changes nothing
    assert o.state_of(d, at(16), "no", False) == "declined"
    assert o.state_of(d, at(17, 15), None, False) == "passed" and o.state_of(d, at(17, 15), "yes", False) == "passed"
    assert o.state_of(d, at(17, 30), "yes", True) == "accepted" and o.state_of(d, at(18, 15), "yes", True) == "passed"


def held(walks, now, state="suggested", fits: o.Fit | None = None, offer: o.Offer | None = None, **gate):
    return o.suggest_held(now, offer or dinner(walks), fits or o.Fit("yes"), state, o.Gate(**gate))


def test_the_suggestion_window_is_ninety_to_thirty_minutes_before(walks):
    assert held(walks, at(15, 44)) == "not within the suggestion window"
    assert held(walks, at(15, 45)) is None and held(walks, at(16, 0)) is None and held(walks, at(16, 45)) is None
    assert held(walks, at(16, 46)) == "not within the suggestion window"
    assert held(walks, at(15, 0), from_min=180) is None and held(walks, at(16, 40), until_min=45) == "not within the suggestion window"


def test_the_quiet_rules(walks):
    assert held(walks, at(16), enabled=False) == "suggestions are switched off"
    assert held(walks, at(16), awake=False) == "outside waking hours"
    assert held(walks, at(16), fixed="COP4533 lecture") == "during COP4533 lecture"
    assert held(walks, at(16), leave_by=at(16, 20), event_start=at(16, 30)) == "close to a time to leave"
    assert held(walks, at(16), leave_by=at(16, 31), event_start=at(16, 40)) is None  # more than half an hour before leaving
    assert held(walks, at(16), last=at(15, 20)) == "something was suggested less than forty-five minutes ago"
    assert held(walks, at(16), last=at(15, 15)) is None
    assert held(walks, at(16), fits=o.Fit("no", reason="it overlaps Lab, which is fixed")) == "does not fit: it overlaps Lab, which is fixed"
    assert held(walks, at(16), fits=o.Fit("nearly")) is None
    nobody = o.parse(item(summary="until 6:15 PM · '63 South · free · optional"), walks.locate, walks.name, TZ)
    assert held(walks, at(16), offer=nobody) == "nobody is named"


def test_once_only_and_never_for_a_declined_accepted_or_passed_one(walks):
    assert held(walks, at(16), said=True) == "already suggested"
    for state in ("declined", "accepted", "passed"):
        assert held(walks, at(16), state=state) == state


# --- what a yes would move ---


def test_changes_lists_only_what_moved_shrank_or_went_and_never_a_fixed_block(walks):
    lab = move("outlook:lab", "class", at(14), 60, place="ucf.eng2", fixed=True, title="Lab")
    old = [lab, move("m0", "study", at(16, 30), 60, title="Study"), move("m1", "eat", at(18, 0), 30, place="ucf.union", title="Eat"),
           move("m2", "study", at(19, 0), 25, title="Study")]
    mine = move(o.block_id(DINNER), "eat", at(17, 15), 60, place="ucf.dining_63", fixed=True, title="Friends' dinner")
    new = [lab, move("m0", "study", at(16, 0), 45, title="Study"), mine, move("m1", "study", at(19, 0), 25, title="Study")]
    rows = o.changes(old, new, at(15, 30), TZ)
    assert [(r["id"], r["change"]) for r in rows] == [("m0", "moved"), ("m1", "dropped")]
    assert rows[0] == {"id": "m0", "title": "Study", "kind": "study", "fixed": False, "change": "moved", "start": "16:30", "end": "17:30",
                       "new_start": "16:00", "new_end": "16:45"}
    assert not [r for r in rows if r["fixed"]]
    assert o.changes(old, old, at(15, 30), TZ) == []


# --- the contract: additive ---


def test_the_contract_additions_are_additive_and_the_answer_is_box_only():
    assert "day.optional.answered" in REGISTRY and "day.optional.answered" in BOX_ONLY_EVENTS
    e = DayOptionalAnswered(ts=at(16, 1), day=at(16).date(), id=DINNER, answer="yes", at=at(16, 1))
    assert e.plan_id is None and e.action_id is None
    assert PlanMove.model_fields["origin"].default is None and DayReplanProposed.model_fields["for_optional"].default is None
    for moment in ("suggest", "start"):
        assert moment in MOMENTS and moment in BOUNDS
        DayNarrated(ts=at(16), day=at(16).date(), moment=moment, spoken="x", display="x", facts_hash="0")


# --- the two new moments in words ---

SUGGEST = {"moment": "suggest", "title": "Friends' dinner", "who": ["Riley", "Sam", "Alex"], "at": "17:15", "until": "18:15", "place": "'63 South",
           "walk_min": 7, "leave_by": "17:08", "fits": "yes", "before": {"title": "your study block", "at": "18:30"}}


def test_a_suggestion_reads_naturally():
    said = render(SUGGEST)
    assert said.spoken == ("Riley, Sam and Alex are having dinner at '63 South at quarter past five. It's a seven-minute walk and it fits "
                           "before your study block. Want me to add it?")
    assert said.display == ("Friends' dinner: 5:15 PM to 6:15 PM at '63 South. 7 min walk, leave by 5:08 PM. "
                            "Fits before your study block at 6:30 PM. Add it?")  # the screen's line (and a push) names nobody


def test_a_suggestion_for_a_host_one_friend_no_walk_and_a_nearly():
    hosted = render({"moment": "suggest", "title": "COP4533 review session", "host": "Andria", "who": ["Maya", "Priya"], "at": "18:30",
                     "until": "20:00", "place": "John C. Hitt Library", "fits": "yes"})
    assert hosted.spoken == ("Andria is hosting COP4533 review session at John C. Hitt Library at half past six, with Maya and Priya. "
                             "It fits the rest of your day. Want me to add it?")
    one = render({"moment": "suggest", "title": "Food truck night", "who": ["Riley"], "at": "18:30", "place": "Memory Mall", "walk_min": 12,
                  "leave_by": "18:18", "fits": "nearly", "moves": ["your study block"]})
    assert one.spoken == ("Riley is going to Food truck night at Memory Mall at half past six. It's a twelve-minute walk, and your study block "
                          "would have to move to make room. Want me to add it?")
    assert "Nearly fits: your study block would move." in one.display
    assert not [n for n in ("Andria", "Maya", "Priya", "Riley") if n in hosted.display + one.display]
    early = render({"moment": "suggest", "title": "Board game night", "who": ["Riley", "Sam"], "at": "20:00", "place": "Student Union",
                    "fits": "nearly", "early_min": 10, "next": "Lab"})
    assert early.spoken == ("Riley and Sam are going to Board game night at Student Union at eight o'clock. You'd leave ten minutes early for Lab. "
                            "Want me to add it?")


def test_the_grounding_check_accepts_a_faithful_rewrite_of_a_suggestion_and_rejects_inventions():
    draft = render(SUGGEST).spoken
    good = "Riley, Sam and Alex are having dinner at '63 South at quarter past five. It's a seven-minute walk, and it fits before your study block. Want me to add it?"
    assert check(good, SUGGEST, draft) == []
    assert any("time that is not in the facts" in p for p in check(good.replace("quarter past five", "half past five"), SUGGEST, draft))
    assert any("name that is not in the facts" in p for p in check(good.replace("Alex", "Jordan"), SUGGEST, draft))
    assert any("number that is not in the facts" in p for p in check(good.replace("seven-minute", "twenty-minute"), SUGGEST, draft))
    assert "dropped the closing question" in check(good.replace(" Want me to add it?", " I'll add it."), SUGGEST, draft)
    assert any("not in the facts: 'lunch'" in p for p in check(good.replace("dinner", "lunch"), SUGGEST, draft))


START = {"moment": "start", "block": "your study block", "kind": "study", "at": "14:00", "min": 25, "place": "the desk", "there": True,
         "deadline": "Project Milestone 1"}


def test_a_block_start_with_the_person_already_there():
    said = render(START)
    assert said.spoken == "It's two o'clock: your study block on Project Milestone 1, twenty-five minutes. You're at the desk already."
    assert said.display == "Your study block on Project Milestone 1, 2 PM, 25 min. You're at the desk already."


def test_a_block_start_with_a_walk_is_said_as_the_time_to_leave():
    said = render({"moment": "start", "block": "lunch", "kind": "eat", "at": "11:30", "min": 30, "place": "Student Union", "walk_min": 7})
    assert said.spoken == "Time for lunch at Student Union; it's a seven-minute walk."
    assert said.display == "Lunch, 11:30 AM, 30 min at Student Union. 7 min walk: leave now."


def test_a_block_start_with_nobody_seen_and_several_short_blocks_said_as_one():
    plain = render({k: v for k, v in START.items() if k != "there"})
    assert plain.spoken == "It's two o'clock: time for your study block on Project Milestone 1, twenty-five minutes, at the desk."
    batch = {"moment": "start", "block": "your study block", "kind": "study", "at": "14:00", "count": 3, "until": "15:45",
             "place": "John C. Hitt Library", "there": True}
    assert render(batch).spoken == "It's two o'clock: three short study blocks from here, through to quarter to four. You're at John C. Hitt Library already."
    assert render(batch).display == "3 short study blocks, 2 PM to 3:45 PM. You're at John C. Hitt Library already."
    walked = render({**batch, "there": False, "walk_min": 5})
    assert walked.spoken == "Time for three short study blocks at John C. Hitt Library, from two o'clock, through to quarter to four; it's a five-minute walk."
    work = render({**batch, "there": False}, Words(activity="work"))
    assert work.spoken == "It's two o'clock: time for three short work blocks from two o'clock, through to quarter to four at John C. Hitt Library."


def test_the_grounding_check_on_a_block_start():
    draft = render(START).spoken
    good = "It's two o'clock, time for your study block on Project Milestone 1: twenty-five minutes. You're at the desk already."
    assert check(good, START, draft) == []
    assert any("time that is not in the facts" in p for p in check(good.replace("two o'clock", "three o'clock"), START, draft))
    assert any("number that is not in the facts" in p for p in check(good.replace("twenty-five", "forty"), START, draft))
    assert any("not in the facts" in p for p in check(good + " You're late.", START, draft))


def test_a_yes_is_narrated_as_the_day_with_the_event_worked_in():
    facts = {"moment": "replan", "asked": True, "added": "Friends' dinner", "before": [{"kind": "study", "count": 1, "min": 25, "at": "16:15"}],
             "first": {"title": "Friends' dinner", "at": "17:15", "place": "'63 South", "walk_min": 16, "leave_by": "16:59"}, "blocks": 3, "ask": "plan"}
    said = render(facts)
    assert said.spoken.startswith("Okay, I've worked Friends' dinner in: ") and "Friends' dinner at quarter past five" in said.spoken
    assert "Leave by five to five for that; it's a sixteen-minute walk." in said.spoken and said.spoken.endswith("Want me to switch to that?")
    assert said.display.startswith("With Friends' dinner added. ")
    gone = render({**facts, "added": None, "removed": "Friends' dinner", "first": None})
    assert gone.spoken.startswith("Okay, Friends' dinner is out: ")
