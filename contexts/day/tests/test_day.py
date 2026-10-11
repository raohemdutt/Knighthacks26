"""The Day loop (D1): the morning plan, the live status, re-plans, the evening review, replay and the API."""

import asyncio
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from bus import EventBus
from contexts.agent.application.memory_port import FakeMemory
from contexts.circle.application.presence import PresenceBoard
from contexts.connectors.application.hub import ConnectorHub
from contexts.connectors.domain.spec import ConnectorSpec
from contexts.day.api import day_router
from contexts.day.application.loop import DayLoop
from contexts.day.domain.choose import choose
from contexts.day.domain.clock import DayClock
from contexts.day.domain.patterns import patterns
from contexts.demo.story import story
from contexts.play.dayrun.service import DayRun
from contexts.play.domain.places import PlaceSet
from contracts import REGISTRY
from contracts.circle import PlaceShared
from contracts.day import DayPlanProposed, DayReplanProposed, DayReviewed, DayStatusChanged
from contracts.experience import ApprovalDecided
from contracts.memory import Episode, Item
from contracts.perception import ZoneEvent
from contracts.workspace import ActionCompleted, ApprovalRequested

TZ = ZoneInfo("America/New_York")
SUNDAY = date(2026, 10, 11)
MILESTONE = "cop4533:assignment:88213"
MENTOR = "outlook:kh-mentor:2026-10-11T09:20:00-04:00"
CLOCK = DayClock(tz=TZ, wake=datetime.min.time().replace(hour=8), bedtime=datetime.min.time().replace(hour=20))


def at(hour: int, minute: int = 0, day: int = 11) -> datetime:
    return datetime(2026, 10, day, hour, minute, tzinfo=TZ)


def never(spec):
    raise AssertionError("a connector was started")


class FakeHabits:
    """What the Day loop reads from Habits: the digest and the history."""

    def __init__(self, sleep: float | None = 5.4, source: str = "watch", meals: int = 1, focus: float = 0.5) -> None:
        self.sleep, self.source, self.meals, self.focus = sleep, source, meals, focus
        self.now = at(9, 30)
        self.days: list[dict] = []

    def digest(self, day=None, now=None) -> dict:
        rows = [{"id": "meals", "value": self.meals, "applies": True},
                {"id": "focus", "value": self.focus, "applies": True}]
        if self.sleep is not None:
            rows.append({"id": "sleep", "value": self.sleep, "applies": True, "source": self.source})
        return {"day": (day or SUNDAY).isoformat(), "habits": rows}

    def history_for(self, habit_id: str, days: int = 7) -> list[dict]:
        return [{"day": d["day"], "value": d[habit_id], "closed": True, "applies": True} for d in self.days if habit_id in d][-days:]


@pytest.fixture(scope="module")
def items() -> list[Item]:
    return [Item(**e.model_dump(include={"connector", "kind", "external_id", "title", "when", "url", "summary"}))
            for e in story() if e.type == "connector.item.synced"]


@pytest.fixture(scope="module")
def places() -> PlaceSet:
    return PlaceSet.load("config/places")


def hub() -> ConnectorHub:
    return ConnectorHub([ConnectorSpec(name="calendar", module="x", writes=["create_event"])], factory=never)


def make(items, places, tmp_path, habits=None, presence=None, with_hub=True, episodes=None) -> DayLoop:
    memory = FakeMemory(items=items, episodes=episodes)
    h = hub() if with_hub else ConnectorHub([], factory=never)
    dayrun = DayRun(places, memory=memory, habits=habits, presence=presence, hub=h, data_dir=tmp_path, clock=lambda: at(9, 30))
    return DayLoop(dayrun, places, habits=habits, memory=memory, presence=presence, hub=h, clock=CLOCK)


def shared(member: str, place: str, ts: datetime) -> PlaceShared:
    return PlaceShared(ts=ts, circle_id="team", member_id=member, place=place, source="phone", expires_at=ts + timedelta(hours=8))


def zone(ts: datetime, zone_id: str, kind: str = "enter", camera: str = "cam-desk") -> ZoneEvent:
    return ZoneEvent(ts=ts, camera_id=camera, track_id=1, label="person", zone_id=zone_id, kind=kind)


def feed(loop: DayLoop, events: list) -> list:
    """Hand events to the loop the way the bus does: every handler sees an event before anyone sees what it caused."""
    p, t = loop.planner, loop.tracker
    out: list = []
    queue = list(events)
    while queue:
        e = queue.pop(0)
        produced: list = []
        if isinstance(e, ZoneEvent):
            t.on_zone_state(e)
            produced += p.on_event(e) or []
            produced += t.on_zone_live(e) or []
        elif isinstance(e, DayPlanProposed):
            p.on_plan_proposed(e)
            produced += t.on_change(e) or []
        elif isinstance(e, DayReplanProposed):
            p.on_replan_proposed(e)
        elif isinstance(e, DayStatusChanged):
            t.on_status_changed(e)
        out += produced
        queue = produced + queue
    return out


# --- the board and the plan ---


def test_the_plan_is_built_from_items_habits_and_presence_and_names_every_source(items, places, tmp_path):
    presence = PresenceBoard("team", "me", places=["Library", "Our table"], opted_in=True)
    presence.on_shared(shared("maya", "Library", at(8)))
    loop = make(items, places, tmp_path, habits=FakeHabits(sleep=5.4, source="watch"), presence=presence)
    events = loop.planner.plan(at(8, 10))
    plan, card = events
    assert isinstance(plan, DayPlanProposed) and plan.day == SUNDAY and plan.action_id == card.action_id
    assert card.kind == "connector_write" and card.target == "calendar.create_event" and card.params["events"]
    by_kind = {(s.kind, s.id): s for s in plan.sources}
    assert by_kind["item", MILESTONE].origin == "canvas" and "Milestone1" in by_kind["item", MILESTONE].text
    assert by_kind["item", MENTOR].origin == "calendar" and "9:20 AM" in by_kind["item", MENTOR].text
    assert by_kind["habit", "sleep"].text == "slept 5.4 h last night" and by_kind["habit", "sleep"].origin == "watch"
    assert by_kind["habit", "meals"].origin == "desk camera"
    assert by_kind["member", "maya"].origin == "circle" and "John C. Hitt Library" in by_kind["member", "maya"].text
    assert ("place", "ucf") in by_kind
    assert plan.missing == []
    # the fixed blocks from the calendar and the chosen moves, in time order, with the editable moves beside them
    fixed = [m.id for m in plan.moves if m.fixed]
    assert MENTOR in fixed and all(a.start <= b.start for a, b in zip(plan.moves, plan.moves[1:]))
    assert plan.dayrun_moves and plan.dayrun_moves[0].slot == "s2"  # the library with Maya, 4 real minutes from the check-in
    assert "Here's your day" in plan.text and "Because:" in plan.text and "slept 5.4 h" in plan.text


def test_missing_sources_are_said_not_hidden(items, places, tmp_path):
    loop = make(items, places, tmp_path, habits=None, presence=None, with_hub=False)
    plan, card = loop.planner.plan(at(8, 10))
    assert "habits: not configured" in plan.missing and "circle: not configured" in plan.missing
    assert "No calendar connector" in card.reason  # the card is still raised; approving commits the plan on the box
    loop = make(items, places, tmp_path, habits=FakeHabits(sleep=6.35, source="camera"))
    plan, _ = loop.planner.plan(at(8, 10))
    assert any(m.startswith("watch:") for m in plan.missing)


def test_the_choice_is_deterministic_and_wins_on_the_demo_day(items, places, tmp_path):
    loop = make(items, places, tmp_path, habits=FakeHabits())
    board = loop.planner.dayrun.board(at(8, 10))
    first = choose(loop.planner.dayrun.cfg, loop.planner.dayrun.walks, board)
    second = choose(loop.planner.dayrun.cfg, loop.planner.dayrun.walks, board)
    assert [m.model_dump() for m in first[0]] == [m.model_dump() for m in second[0]]
    assert first[1].outcome.won and first[1].stars >= 2 and not first[1].rejected
    kinds = {m.kind for m in first[0]}
    assert {"study", "eat", "sleep"} <= kinds  # study near the deadline, meals, and the night's sleep
    assert first[1].totals.sleep_hours >= 7.5 and first[1].totals.meals >= 2


def test_once_per_day_at_wake_and_when_the_box_starts_after_wake(items, places, tmp_path):
    loop = make(items, places, tmp_path, habits=FakeHabits())
    p = loop.planner
    assert p.tick(at(7, 30)) == []  # before wake: nothing
    quiet = make(items, places, tmp_path / "quiet", habits=None)
    assert quiet.planner.tick(at(8, 0)) == [] and quiet.planner.on_event(zone(at(8, 1), "desk"))  # a tick needs a sign of life today
    events = p.tick(at(8, 0))  # FakeHabits has seen 9:30 today
    assert [type(e) for e in events] == [DayPlanProposed, type(events[1])] and events[0].plan_id == f"plan-{SUNDAY}-0800"
    p.on_plan_proposed(events[0])
    assert p.tick(at(9, 0)) == [] and p.on_event(zone(at(9, 5), "desk")) == []  # once a day
    # a box that comes up at three in the afternoon with no plan for today plans then, from that moment
    late = make(items, places, tmp_path / "late", habits=FakeHabits())
    plan = late.planner.tick(at(15, 0))[0]
    assert plan.start == at(15, 0) and plan.plan_id == f"plan-{SUNDAY}-1500"
    # ... but not after bedtime: the day is over
    over = make(items, places, tmp_path / "over", habits=FakeHabits())
    assert over.planner.tick(at(21, 0)) == [] and SUNDAY in over.planner.skipped


def test_a_remake_and_the_persons_own_moves_replace_the_plan(items, places, tmp_path):
    loop = make(items, places, tmp_path, habits=FakeHabits())
    p = loop.planner
    first = p.plan(at(8, 10))[0]
    p.on_plan_proposed(first)
    again = p.plan(at(8, 10), remake=True)
    assert again[0].plan_id != first.plan_id and "Replace" in again[1].reason
    from contexts.play.dayrun.models import Move

    mine = p.edited(at(8, 10), [Move(slot="s1", kind="eat", place="apartment.kitchen", minutes=30, start=at(8, 15))])
    assert mine[0].by == "person" and "Your edited day" in mine[0].text
    assert [m.kind for m in mine[0].dayrun_moves] == ["eat"] and [m.title for m in mine[0].moves if not m.fixed] == ["Eat at Kitchen"]


# --- the live day ---


def test_drift_is_seen_and_an_infeasible_next_event_triggers_one_replan(items, places, tmp_path):
    loop = make(items, places, tmp_path, habits=FakeHabits(meals=0))  # nothing eaten yet: the plan opens with breakfast
    out = feed(loop, [zone(at(8, 10), "desk")])
    plan = next(e for e in out if isinstance(e, DayPlanProposed))
    statuses = [e for e in out if isinstance(e, DayStatusChanged)]
    assert statuses and statuses[-1].seen_at == "apartment.desk" and statuses[-1].seen_how == "desk camera"
    assert statuses[-1].next_event.id == MENTOR and statuses[-1].next_event.walk_minutes == 10
    assert statuses[-1].next_event.leave_by == at(9, 10) and statuses[-1].risk == "ok"
    # breakfast at the kitchen is the plan at 8:35; the desk is where the cameras see you: a few minutes behind
    out = feed(loop, [zone(at(8, 35), "desk")])
    s = next(e for e in out if isinstance(e, DayStatusChanged))
    assert s.block_title == "Eat at Kitchen" and s.should_be == "apartment.kitchen" and s.drift_minutes > 0
    # out the door at 9:05: on the way, "leave now"
    out = feed(loop, [zone(at(9, 5), "door", "exit", camera="cam-door")])
    s = next(e for e in out if isinstance(e, DayStatusChanged))
    assert s.seen_at == "apartment.door" and s.risk == "leave_now"
    # back at the desk at 9:12: the 9:20 check-in is 10 minutes away, so it is infeasible, and one re-plan is proposed
    out = feed(loop, [zone(at(9, 12), "desk")])
    s = next(e for e in out if isinstance(e, DayStatusChanged))
    assert s.risk == "infeasible" and s.reason == "you will miss Knight Hacks: mentor check-in"
    replans = [e for e in out if isinstance(e, DayReplanProposed)]
    assert len(replans) == 1 and replans[0].replaces == plan.plan_id and replans[0].for_event == MENTOR
    assert "At Desk at 09:12" in replans[0].why and "10 minute walk" in replans[0].why
    card = next(e for e in out if e.type == "agent.action.proposed")
    assert card.action_id == replans[0].action_id and "Replace today's plan with" in card.reason
    # while it is pending, no second one; after it expired, drift still there, once more; then no more for that event
    assert feed(loop, [zone(at(9, 14), "desk")]) == [] or not any(isinstance(e, DayReplanProposed) for e in feed(loop, [zone(at(9, 15), "desk")]))
    loop.planner.on_action_completed(ActionCompleted(ts=at(9, 17), action_id=card.action_id, status="expired"))
    out = feed(loop, [zone(at(9, 30), "desk")])
    assert sum(isinstance(e, DayReplanProposed) for e in out) == 1
    loop.planner.on_action_completed(ActionCompleted(ts=at(9, 35), action_id=out[-1].action_id, status="expired"))
    assert not any(isinstance(e, DayReplanProposed) for e in feed(loop, [zone(at(9, 50), "desk")]))


def test_statuses_come_at_most_once_a_minute_and_only_on_a_material_change(items, places, tmp_path):
    loop = make(items, places, tmp_path, habits=FakeHabits())
    feed(loop, [zone(at(8, 10), "desk")])
    assert feed(loop, [zone(at(8, 10).replace(second=30), "desk", "dwell")]) == []  # nothing changed
    out = feed(loop, [zone(at(8, 10).replace(second=40), "door", camera="cam-door")])  # changed, but within the minute: held
    assert out == [] and loop.tracker.dirty
    out = loop.tracker.tick(at(8, 12))
    assert len(out) == 1 and out[0].seen_at == "apartment.door" and out[0].reason == "seen at Front door"


def test_approval_commits_the_plan_and_an_approved_replan_becomes_the_days_plan(items, places, tmp_path):
    loop = make(items, places, tmp_path, habits=FakeHabits())
    p = loop.planner
    plan, card = p.plan(at(8, 10))
    p.on_plan_proposed(plan)
    p.on_approval_requested(ApprovalRequested(ts=at(8, 10), action_id=card.action_id, kind=card.kind, target=card.target,
                                              reason=card.reason, expires_at=at(8, 15)))
    assert p.on_decided(ApprovalDecided(ts=at(8, 16), action_id=card.action_id, decision="approve")) == []  # too late
    (committed,) = p.on_decided(ApprovalDecided(ts=at(8, 12), action_id=card.action_id, decision="approve"))
    assert committed.plan_id == plan.plan_id and committed.event_id == f"day-committed-{plan.plan_id}"
    p.on_plan_committed(committed)
    assert plan.plan_id in p.committed and p.on_decided(ApprovalDecided(ts=at(8, 13), action_id=card.action_id, decision="approve")) == []
    replan, rcard = p.replan(at(9, 12), "apartment.desk", "why", for_event=MENTOR)
    p.on_replan_proposed(replan)
    assert p.pending[SUNDAY].plan_id == replan.plan_id and p.current[SUNDAY].plan_id == plan.plan_id
    (c2,) = p.on_decided(ApprovalDecided(ts=at(9, 13), action_id=rcard.action_id, decision="approve"))
    p.on_plan_committed(c2)
    assert p.current[SUNDAY].plan_id == replan.plan_id and SUNDAY not in p.pending
    # a denial clears a pending re-plan and holds the next one back
    r2, c3 = p.replan(at(9, 40), "apartment.desk", "why")
    p.on_replan_proposed(r2)
    p.on_action_completed(ActionCompleted(ts=at(9, 41), action_id=c3.action_id, status="denied"))
    assert SUNDAY not in p.pending and not p.can_replan(at(9, 50), timedelta(minutes=15)) and p.can_replan(at(10), timedelta(minutes=15))


# --- replay ---


def test_a_restart_mid_day_rebuilds_the_same_plan_and_status_from_the_log(items, places, tmp_path):
    async def live() -> tuple:
        bus = EventBus(tmp_path / "events")
        loop = make(items, places, tmp_path / "a", habits=FakeHabits())
        loop.subscribe(bus)
        for e in (zone(at(8, 10), "desk"), zone(at(8, 35), "desk"), zone(at(9, 5), "door", "exit", camera="cam-door"), zone(at(9, 12), "desk")):
            await bus.publish(e)
        await bus.publish_all(loop.tick(at(9, 30)))
        return loop

    async def restarted() -> DayLoop:
        bus = EventBus(tmp_path / "events")
        loop = make(items, places, tmp_path / "b", habits=FakeHabits())
        loop.subscribe(bus)
        await bus.replay_into()
        return loop

    before = asyncio.run(live())
    after = asyncio.run(restarted())
    assert after.tracker.status == before.tracker.status and after.tracker.status.risk in ("infeasible", "late")
    assert after.tracker.seen == before.tracker.seen
    assert after.planner.current[SUNDAY] == before.planner.current[SUNDAY]
    assert after.planner.pending[SUNDAY] == before.planner.pending[SUNDAY]
    assert after.planner.cards == before.planner.cards and after.planner.replans_for == before.planner.replans_for
    # replay proposed nothing: the log holds exactly what the live run wrote
    types = [e.type for e in EventBus(tmp_path / "events").read_log()]
    assert types.count("day.plan.proposed") == 1 and types.count("day.replan.proposed") == 1
    assert after.planner.tick(at(9, 31)) == []  # and the restarted box does not plan again


# --- the evening review ---


def test_the_evening_review_scores_the_day_and_the_real_days_stars(items, places, tmp_path):
    habits = FakeHabits(sleep=6.35, source="camera", meals=2, focus=2.5)
    loop = make(items, places, tmp_path, habits=habits)
    p = loop.planner
    plan = p.plan(at(8, 10))[0]
    p.on_plan_proposed(plan)
    assert p.tick(at(19, 59)) == []  # not bedtime yet
    (review,) = p.tick(at(20, 0))
    assert isinstance(review, DayReviewed) and review.day == SUNDAY and review.plan_id == plan.plan_id
    assert review.score is not None and 0 <= review.score <= 100
    assert "You followed" in review.text and "by Today's Plan's rules" in review.text
    assert review.plan_stars == plan.outcome.stars and review.points > 0 and review.streak == (1 if review.stars else 0)
    assert review.change_tomorrow
    p.on_reviewed(review)
    assert p.tick(at(20, 30)) == []  # once
    # tomorrow's plan carries it as a source
    monday = make(items, places, tmp_path / "mon", habits=habits)
    monday.planner.reviews[SUNDAY] = review
    nxt = monday.planner.plan(at(8, 5, day=12))[0]
    assert any(s.kind == "review" and s.id == plan.plan_id for s in nxt.sources)


def test_the_review_also_runs_when_habits_closes_the_day(items, places, tmp_path):
    from contracts.habits import DayClosed

    loop = make(items, places, tmp_path, habits=FakeHabits())
    plan = loop.planner.plan(at(8, 10))[0]
    loop.planner.on_plan_proposed(plan)
    out = loop.planner.on_day_closed(DayClosed(ts=at(0, 2, day=12), habit_id="sleep", day=SUNDAY, met=True, value=7, streak=1))
    assert len(out) == 1 and isinstance(out[0], DayReviewed)


# --- the contract ---


def test_no_day_event_carries_anything_beyond_its_typed_fields(items, places, tmp_path):
    loop = make(items, places, tmp_path, habits=FakeHabits())
    out = feed(loop, [zone(at(8, 10), "desk"), zone(at(9, 12), "desk")])
    out += loop.planner.tick(at(20))
    days = [e for e in out if e.type.startswith("day.")]
    assert {e.type for e in days} >= {"day.plan.proposed", "day.status.changed", "day.replan.proposed", "day.reviewed"}
    for e in days:
        cls = REGISTRY[e.type]
        assert type(e) is cls and set(e.model_dump()) == set(cls.model_fields)
        text = e.model_dump_json().lower()
        assert "frame" not in text and "heart" not in text and "hrv" not in text


# --- the API ---


def test_the_api_shapes(items, places, tmp_path):
    episodes = [Episode(id="e1", camera_id="cam-desk", start=at(9, 12), end=at(9, 14), labels=["person"], captions=["Two people rehearse."],
                        keyframe_refs=["frames:cam-desk:1"])]
    loop = make(items, places, tmp_path, habits=FakeHabits(), episodes=episodes)
    published = []

    async def publish(e):
        published.append(e)
        for handler in (loop.planner.on_plan_proposed, loop.planner.on_replan_proposed):
            if isinstance(e, (DayPlanProposed, DayReplanProposed)) and handler.__name__.split("_")[1] == e.type.split(".")[1]:
                handler(e)

    app = FastAPI()
    app.include_router(day_router(loop, publish=publish, memory=loop.memory, clock=lambda: at(9, 30)))
    c = TestClient(app)

    now = c.get("/api/day/now").json()
    assert now["day"] == "2026-10-11" and now["plan"] is None and now["status"] is None and now["friends"] == []
    assert now["last_seen"]["camera_id"] == "cam-desk" and now["last_seen"]["caption"] == "Two people rehearse."
    assert c.get("/api/day/plan").status_code == 404

    made = c.post("/api/day/plan", json={}).json()
    assert made["type"] == "day.plan.proposed" and made["action_id"] and [e.type for e in published] == ["day.plan.proposed", "agent.action.proposed"]
    plan = c.get("/api/day/plan", params={"day": "2026-10-11"}).json()
    assert plan["plan_id"] == made["plan_id"] and plan["committed"] is False
    assert plan["moves"] and set(plan["moves"][0]) == {"slot", "kind", "place", "minutes", "start", "with_friend", "deadline", "event"}
    assert plan["blocks"][0]["title"] and plan["outcome"]["won"] is not None
    now = c.get("/api/day/now").json()
    assert now["plan"]["plan_id"] == made["plan_id"] and now["committed"] is False and now["replan"] is None

    edited = c.post("/api/day/plan", json={"moves": [{"slot": "s1", "kind": "eat", "place": "apartment.kitchen", "minutes": 30}]}).json()
    assert edited["plan"]["by"] == "person" and edited["plan"]["dayrun_moves"] == [
        {"slot": "s1", "kind": "eat", "place": "apartment.kitchen", "minutes": 30, "start": None, "with_friend": None, "deadline": None, "event": None}]

    loop.tracker.on_zone_state(zone(at(9, 12), "desk"))
    re = c.post("/api/day/plan", json={"from_here": True}).json()
    assert re["type"] == "day.replan.proposed" and re["plan"]["asked"] is True and "At Desk" in re["plan"]["why"]
    now = c.get("/api/day/now").json()
    assert now["replan"]["plan_id"] == re["plan_id"]
    assert c.get("/api/day/patterns").json() == {"patterns": []}


def test_patterns_are_counted_statements_or_nothing():
    h = FakeHabits()
    assert patterns(h) == []
    h.days = [{"day": f"2026-10-{d:02d}", "sleep": 5 if d % 2 else 7.5, "focus": 2 if d % 2 else 4, "meals": 1 if d % 2 else 2} for d in range(1, 9)]
    out = patterns(h)
    assert out[0]["text"].startswith("On the 4 days after a night under 6 h, desk time (h) averaged 2.0 against 4.0 on the other 4.")
    assert any("of the last 8 measured nights were under 6 h" in p["text"] for p in out)
