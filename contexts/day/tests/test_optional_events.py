"""Optional events you can say yes or no to (contexts/day/application/optional.py), through the real Day loop, the
real bus and the real approvals: the list with fit, walk, leave-by and state; a yes that only ever raises a re-plan
behind the usual card; the event as an ordinary fixed block once that card is approved (next event, leave-by,
leave now, review); a no; the same state on a second box built from the same log; and what never happens (a fixed
event moved, a friend's name in a plan block)."""

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from urllib.parse import quote
from zoneinfo import ZoneInfo

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from bus import EventBus
from contexts.agent.application.memory_port import FakeMemory
from contexts.connectors.application.hub import ConnectorHub
from contexts.connectors.domain.spec import ConnectorSpec
from contexts.day.api import day_router
from contexts.day.application.loop import DayLoop
from contexts.day.application.optional import OptionalEvents, Refused
from contexts.day.domain import optional as rule
from contexts.day.tests.test_day import CLOCK, never, places, zone  # noqa: F401
from contexts.demo.nearby import nearby_events
from contexts.demo.seed import settle
from contexts.play.dayrun.agreed import AgreedPlans
from contexts.play.dayrun.service import DayRun
from contexts.workspace.domain.approvals import Approvals
from contracts.circle import EventAgreed
from contracts.day import DayNarrated, DayOptionalAnswered, DayPlanCommitted, DayReplanProposed, DayReviewed, DayStatusChanged
from contracts.experience import ApprovalDecided
from contracts.memory import Item
from contracts.workspace import ActionCompleted

TZ = ZoneInfo("America/New_York")
MONDAY = datetime(2026, 10, 12, tzinfo=TZ).date()
DINNER = "nearby:friends-dinner:2026-10-12T17:15:00-04:00"
DINNER_BLOCK = "optional:friends-dinner:2026-10-12T17:15:00-04:00"
REVIEW = "nearby:cop4533-review:2026-10-12T18:30:00-04:00"
SOCCER = "nearby:pickup-soccer:2026-10-13T16:30:00-04:00"
LECTURE = "outlook:cop4533-lecture:2026-10-12T10:40:00-04:00"
FRIENDS = ("Riley", "Sam", "Alex", "Andria", "Maya", "Priya", "Jordan")


def at(hour: int, minute: int = 0, day: int = 12) -> datetime:
    return datetime(2026, 10, day, hour, minute, tzinfo=TZ)


def calendar() -> list[Item]:
    """Monday's own calendar: one lecture, and an assignment due the next night."""
    return [Item(connector="calendar", kind="calendar_event", external_id=LECTURE, title="COP4533 lecture", when=at(10, 40),
                 summary="until 11:55 AM · Classroom Building I"),
            Item(connector="canvas", kind="assignment", external_id="cop4533:assignment:88213", title="Project - Milestone1 Fall2026 [COP4533]",
                 when=at(23, 59, 13), summary="Dynamic programming: code plus a runtime analysis writeup.")]


def near() -> list[Item]:
    """The seven seeded "Near you" events as Memory holds them (fixtures/demo/nearby.yaml, the story ending Sunday morning)."""
    return [Item(**e.model_dump(include={"connector", "kind", "external_id", "title", "when", "url", "summary"}))
            for e in nearby_events(at(8, 0, 11)) if e.kind == "calendar_event"]


def offer(slug: str, start: datetime, until: str, place: str, tags: str = "with Maya · optional · via GroupMe", title: str | None = None) -> Item:
    return Item(connector="calendar", kind="calendar_event", external_id=f"nearby:{slug}:{start.isoformat()}",
                title=title or slug.replace("-", " ").capitalize(), when=start, summary=f"until {until} · {place} · {tags}")


@dataclass
class Box:
    loop: DayLoop
    bus: EventBus
    out: list  # every event published live on this box
    approvals: Approvals
    dayrun: DayRun
    clock: dict

    @property
    def optional(self) -> OptionalEvents:
        return self.loop.optional

    def of(self, kind) -> list:
        return [e for e in self.out if isinstance(e, kind)]

    def client(self) -> TestClient:
        app = FastAPI()
        app.include_router(day_router(self.loop, publish=self.bus.publish, memory=self.loop.memory, clock=lambda: self.clock["now"]))
        return TestClient(app)

    def publish(self, *events) -> None:
        async def run() -> None:
            await self.bus.publish_all(events)
            await settle()  # a narration the model is asked about comes back from a background task

        asyncio.run(run())

    def morning(self, now: datetime = at(8, 10)) -> None:
        """The morning plan (its card is left to expire, as on a box nobody looked at) and a person at the desk."""
        self.publish(*self.loop.planner.plan(now))

    def tick(self, now: datetime) -> None:
        """One turn of the gateway's ticker at `now`: cards that ran out expire, then the Day loop looks at the day."""
        self.clock["now"] = now

        async def run() -> None:
            await self.bus.publish_all(self.approvals.expire(now))
            await self.bus.publish_all(self.loop.tick(now))
            await settle()

        asyncio.run(run())

    def approve(self, action_id: str, now: datetime) -> None:
        self.clock["now"] = now
        self.publish(ApprovalDecided(ts=now, action_id=action_id, decision="approve"))

    def plan(self):
        return self.loop.planner.current[MONDAY]

    def row(self, key: str, now: datetime) -> dict:
        return next(r for r in self.optional.view(now)["events"] if key in (r["id"], r["slug"]))


def box(places, tmp_path, name: str = "a", items: list[Item] | None = None, writes: bool = False, agreed=None,  # noqa: F811
        day: dict | None = None, runtime=None) -> Box:
    """A Day loop on a bus with the approvals the gateway wires (contexts/gateway/bootstrap.py), and nothing else.
    `day` is agent.yaml's `day:` block; by default Atlas does not bring optional events up by itself here (the
    suggestion has its own tests: test_optional_suggest.py), so these tests are about the answers alone."""
    memory = FakeMemory(items=calendar() + near() if items is None else items)
    hub = ConnectorHub([ConnectorSpec(name="calendar", module="x", writes=["create_event"])] if writes else [], factory=never)
    dayrun = DayRun(places, memory=memory, habits=None, hub=hub, data_dir=tmp_path / name, clock=lambda: at(16), agreed=agreed)
    loop = DayLoop(dayrun, places, memory=memory, hub=hub, clock=CLOCK, runtime=runtime,
                   narration={"suggest_optional": False} if day is None else day)
    bus = EventBus(tmp_path / "events")
    approvals = Approvals({"connector_write": lambda request: "kept as today's plan"})
    bus.subscribe("agent.action.proposed", approvals.on_proposed, replay=True)
    bus.subscribe("workspace.action.completed", approvals.on_completed, replay=True)
    loop.subscribe(bus)
    bus.subscribe("experience.approval.decided", approvals.on_decided)
    out: list = []
    bus.subscribe("", out.append)
    return Box(loop, bus, out, approvals, dayrun, {"now": at(16)})


def afternoon(places, tmp_path, **kw) -> Box:  # noqa: F811
    """Monday at 4 PM: the morning plan is the day's plan, and the desk camera saw someone sit down ten minutes ago."""
    b = box(places, tmp_path, **kw)
    b.morning()
    b.publish(zone(at(15, 50), "desk"))
    b.tick(at(16))
    return b


def said_yes(b: Box, key: str = DINNER, now: datetime = at(16, 1)) -> dict:
    b.clock["now"] = now
    events, body = b.optional.answer(now, key, "yes")
    b.publish(*events)
    return body


def accepted(places, tmp_path, **kw) -> Box:  # noqa: F811
    """... and the person said yes to the friends' dinner at 4:01 PM and approved the card at 4:02 PM."""
    b = afternoon(places, tmp_path, **kw)
    body = said_yes(b)
    b.approve(body["action_id"], at(16, 2))
    return b


def names_in(text: str) -> list[str]:
    return [n for n in FRIENDS if n in text]


# ---------- the list ----------


def test_the_list_has_each_event_with_who_is_going_its_fit_the_walk_and_its_state(places, tmp_path):  # noqa: F811
    b = afternoon(places, tmp_path)
    view = b.optional.view(at(16))
    assert view["now"] == at(16).isoformat() and view["plan_id"] == b.plan().plan_id
    assert [r["slug"] for r in view["events"]] == ["resume-workshop", "friends-dinner", "cop4533-review", "pickup-soccer", "food-truck-night"]
    dinner = view["events"][1]
    assert dinner == {
        "id": DINNER, "slug": "friends-dinner", "title": "Friends' dinner", "start": at(17, 15).isoformat(), "end": at(18, 15).isoformat(),
        "place": "ucf.dining_63", "place_name": "'63 South", "tags": ["with Riley, Sam, Alex", "optional", "via GroupMe"],
        "going": ["Riley", "Sam", "Alex"], "host": None, "source": "GroupMe", "state": "suggested", "answer": None,
        "awaiting_approval": False, "action_id": None, "plan_id": None, "suggested_at": None, "suggest_held": "suggestions are switched off",
        "fit": {"fits": "yes", "reason": "", "walk_minutes": 16, "leave_by": at(16, 59).isoformat(), "from_place": "apartment.desk",
                "displaces": [], "conflicts": [], "leave_early_minutes": 0}}
    # the workshop began at half past twelve: it is over; the review would push the evening's blocks aside
    assert view["events"][0]["state"] == "passed"
    review = view["events"][2]
    assert review["host"] == "Andria" and review["going"] == ["Maya", "Priya"] and review["state"] == "suggested"
    assert review["fit"]["fits"] == "nearly" and review["fit"]["reason"] == "Eat at Kitchen would move"
    assert [d["title"] for d in review["fit"]["displaces"]] == ["Eat at Kitchen", "Study at Desk"] and review["fit"]["conflicts"] == []
    assert all(d["overlap_minutes"] > 0 and d["start"] < d["end"] for d in review["fit"]["displaces"])
    assert json.loads(json.dumps(view)) == view  # plain JSON all the way down


def test_with_no_plan_and_no_memory_the_list_is_still_an_answer(places, tmp_path):  # noqa: F811
    b = box(places, tmp_path)
    view = b.optional.view(at(16))
    assert view["plan_id"] is None and len(view["events"]) == 5 and view["events"][1]["fit"]["walk_minutes"] == 0
    b.optional.memory = None
    assert b.optional.view(at(16)) == {"now": at(16).isoformat(), "plan_id": None, "events": []}


def test_an_offer_is_found_by_its_id_or_by_its_slug(places, tmp_path):  # noqa: F811
    b = afternoon(places, tmp_path)
    assert b.optional.find(DINNER, at(16)).id == b.optional.find("friends-dinner", at(16)).id == DINNER
    assert b.optional.find("nope", at(16)) is None and b.optional.find("nearby:friends-dinner:2026-10-19T17:15:00-04:00", at(16)) is None


# ---------- the acceptance walk, through the API ----------


def test_a_yes_raises_a_replan_behind_the_usual_card_and_approved_it_is_a_block_of_the_day(places, tmp_path):  # noqa: F811
    b = afternoon(places, tmp_path)
    c = b.client()
    morning = b.plan()

    listed = c.get("/api/day/optional").json()
    dinner = next(r for r in listed["events"] if r["id"] == DINNER)
    assert (dinner["state"], dinner["fit"]["fits"], dinner["awaiting_approval"]) == ("suggested", "yes", False)

    # yes: an answer, then a re-plan and its card. The plan itself has not moved.
    b.clock["now"] = at(16, 1)
    r = c.post(f"/api/day/optional/{quote(DINNER, safe='')}", json={"answer": "yes"})
    assert r.status_code == 200, r.text
    yes = r.json()
    replan = b.loop.planner.pending[MONDAY]
    [card] = b.approvals.pending()
    assert (yes["id"], yes["answer"], yes["state"], yes["proposed"]) == (DINNER, "yes", "suggested", True)
    assert (yes["plan_id"], yes["action_id"]) == (replan.plan_id, replan.action_id) == (replan.plan_id, card.id)
    assert yes["block"]["id"] == DINNER_BLOCK and yes["block"]["origin"] == "optional" and yes["block"]["fixed"] is True
    assert yes["moves"] and all(m["change"] in ("moved", "shortened", "lengthened", "dropped") and m["fixed"] is False for m in yes["moves"])
    assert {m["title"] for m in yes["moves"]} <= {m.title for m in morning.moves if not m.fixed}  # only blocks Atlas had chosen
    assert all(a["start"] >= "16:00" for a in yes["adds"]) and yes["fit"]["walk_minutes"] == 16
    assert [e.type for e in b.out[-6:]] == ["day.optional.answered", "day.replan.proposed", "day.narrated", "agent.action.proposed",
                                            "workspace.approval.requested", "day.narrated"]
    [answer] = b.of(DayOptionalAnswered)
    assert (answer.id, answer.answer, answer.plan_id, answer.action_id, answer.at) == (DINNER, "yes", replan.plan_id, card.id, at(16, 1))
    assert replan.for_optional == DINNER and replan.asked and replan.replaces == morning.plan_id
    plan = c.get("/api/day/plan").json()
    assert plan["plan_id"] == morning.plan_id and not [x for x in plan["blocks"] if x["origin"]] and not b.optional.accepted
    now = c.get("/api/day/now").json()
    assert now["plan"]["plan_id"] == morning.plan_id and now["replan"]["plan_id"] == yes["plan_id"] and now["replan"]["for_optional"] == DINNER
    waiting = next(r for r in c.get("/api/day/optional").json()["events"] if r["id"] == DINNER)
    assert (waiting["state"], waiting["answer"], waiting["awaiting_approval"], waiting["action_id"]) == ("suggested", "yes", True, card.id)
    assert DINNER_BLOCK not in [x.id for x in b.dayrun.board(at(16, 1)).blocks]  # not on the board yet

    # approved: the re-plan is the day's plan, and the dinner is an ordinary fixed block of it
    b.approve(card.id, at(16, 2))
    plan = c.get("/api/day/plan").json()
    assert plan["plan_id"] == yes["plan_id"] and plan["committed"] is True
    [block] = [x for x in plan["blocks"] if x["origin"] == "optional"]
    assert block == yes["block"] and (block["title"], block["place"], block["place_name"]) == ("Friends' dinner", "ucf.dining_63", "'63 South")
    assert (block["start"], block["end"], block["kind"]) == (at(17, 15).isoformat(), at(18, 15).isoformat(), "eat")
    b.tick(at(16, 3))
    nxt = c.get("/api/day/now").json()["status"]["next_event"]
    assert (nxt["id"], nxt["walk_minutes"], nxt["leave_by"]) == (DINNER_BLOCK, 16, at(16, 59).isoformat())
    done = next(r for r in c.get("/api/day/optional").json()["events"] if r["id"] == DINNER)
    assert (done["state"], done["awaiting_approval"], done["action_id"]) == ("accepted", False, None)
    assert DINNER_BLOCK in [x.id for x in b.dayrun.board(at(16, 3)).blocks]  # the Day Run board carries it from here on

    # a second box built on the same log shows the same state and says nothing
    again = box(places, tmp_path, name="b")
    assert asyncio.run(again.bus.replay_into()) > 0 and again.out == []
    assert again.optional.accepted == b.optional.accepted == {DINNER} and again.optional.answers == b.optional.answers
    assert again.plan() == b.plan() and again.loop.tracker.status == b.loop.tracker.status
    assert again.optional.view(at(16, 3)) == b.optional.view(at(16, 3))
    assert DINNER_BLOCK in [x.id for x in again.dayrun.board(at(16, 3)).blocks]

    # a no: recorded, declined, nothing proposed
    b.clock["now"] = at(16, 4)
    no = c.post("/api/day/optional/cop4533-review", json={"answer": "no"}).json()
    assert no == {"id": REVIEW, "answer": "no", "state": "declined", "proposed": False}
    declined = next(r for r in c.get("/api/day/optional").json()["events"] if r["id"] == REVIEW)
    assert (declined["state"], declined["answer"], declined["awaiting_approval"]) == ("declined", "no", False)
    assert MONDAY not in b.loop.planner.pending and b.plan().plan_id == yes["plan_id"]
    assert rule.suggest_held(at(17, 30), b.optional.find(REVIEW, at(17, 30)), b.optional.fit(b.optional.find(REVIEW, at(17, 30)), at(17, 30)),
                             b.optional.state(b.optional.find(REVIEW, at(17, 30)), at(17, 30)), rule.Gate()) == "declined"


def test_what_the_api_refuses_and_how(places, tmp_path):  # noqa: F811
    lunch = offer("lab-lunch", at(11, 0), "12:00 PM", "Student Union")  # the lecture runs until 11:55
    b = box(places, tmp_path, items=calendar() + near() + [lunch])
    c = b.client()
    b.clock["now"] = at(8, 20)
    assert c.post("/api/day/optional/nope", json={"answer": "yes"}).status_code == 404
    assert c.post("/api/day/optional/friends-dinner", json={"answer": "maybe"}).status_code == 422
    r = c.post("/api/day/optional/friends-dinner", json={"answer": "yes"})
    assert r.status_code == 409 and r.json()["detail"] == "there is no plan for today yet to work it into; make one first"
    b.morning()
    r = c.post("/api/day/optional/lab-lunch", json={"answer": "yes"})
    assert r.status_code == 409 and r.json()["detail"] == "Lab lunch does not fit: it overlaps COP4533 lecture, which is fixed"
    b.clock["now"] = at(13)
    r = c.post("/api/day/optional/resume-workshop", json={"answer": "yes"})
    assert r.status_code == 409 and r.json()["detail"] == "Resume workshop has already started"
    assert b.of(DayOptionalAnswered) == [] and b.of(DayReplanProposed) == []  # a refusal publishes nothing
    # more than a day ahead: the yes is kept, nothing is proposed, and it is still on offer
    far = c.post(f"/api/day/optional/{quote(SOCCER, safe='')}", json={"answer": "yes"}).json()
    assert (far["proposed"], far["state"]) == (False, "suggested") and "more than a day ahead" in far["reason"]
    [kept] = b.of(DayOptionalAnswered)
    assert (kept.id, kept.answer, kept.plan_id, kept.action_id) == (SOCCER, "yes", None, None) and b.of(DayReplanProposed) == []
    assert b.row(SOCCER, at(13))["state"] == "suggested" and b.row(SOCCER, at(13))["answer"] == "yes"


def test_writes_need_the_guard_and_reads_do_not(places, tmp_path):  # noqa: F811
    from fastapi import HTTPException

    def locked() -> None:
        raise HTTPException(401, "no token")

    b = afternoon(places, tmp_path)
    app = FastAPI()
    app.include_router(day_router(b.loop, publish=b.bus.publish, guard=locked, clock=lambda: at(16)))
    c = TestClient(app)
    assert c.get("/api/day/optional").status_code == 200
    assert c.post("/api/day/optional/friends-dinner", json={"answer": "yes"}).status_code == 401 and b.of(DayOptionalAnswered) == []


# ---------- a yes never changes the plan by itself ----------


def test_a_card_nobody_approves_changes_nothing_and_the_event_is_still_on_offer(places, tmp_path):  # noqa: F811
    b = afternoon(places, tmp_path)
    morning = b.plan()
    body = said_yes(b)
    assert b.row(DINNER, at(16, 5))["awaiting_approval"] is True
    assert b.row(DINNER, at(16, 6))["awaiting_approval"] is False  # five minutes: the card is over, also before the ticker says so
    b.tick(at(16, 7))
    assert [e.status for e in b.of(ActionCompleted) if e.action_id == body["action_id"]] == ["expired"]
    assert b.plan() == morning and MONDAY not in b.loop.planner.pending and b.optional.accepted == set()
    row = b.row(DINNER, at(16, 7))
    assert (row["state"], row["answer"], row["awaiting_approval"]) == ("suggested", "yes", False)
    assert b.dayrun.agreed is b.optional and DINNER_BLOCK not in [x.id for x in b.dayrun.board(at(16, 7)).blocks]
    # an approval that comes too late commits nothing either
    b.approve(body["action_id"], at(16, 8))
    assert b.of(DayPlanCommitted) == [] and b.optional.accepted == set()


def test_a_card_approved_after_a_newer_plan_replaced_it_accepts_nothing(places, tmp_path):  # noqa: F811
    b = afternoon(places, tmp_path)
    c = b.client()
    first = said_yes(b)
    b.clock["now"] = at(16, 2)
    newer = c.post("/api/day/plan", json={"from_here": True}).json()  # the person asked for another re-plan: it has no dinner
    assert newer["plan_id"] != first["plan_id"] and b.row(DINNER, at(16, 2))["awaiting_approval"] is False
    b.approve(first["action_id"], at(16, 3))
    assert [e.plan_id for e in b.of(DayPlanCommitted)] == [first["plan_id"]]  # the planner records it, and the day's plan stays
    assert b.plan().plan_id != first["plan_id"] and b.optional.accepted == set() and b.row(DINNER, at(16, 3))["state"] == "suggested"


def test_before_any_yes_day_runs_port_is_untouched_and_afterwards_whatever_was_there_is_still_asked(places, tmp_path):  # noqa: F811
    alone = afternoon(places, tmp_path / "alone")
    assert alone.dayrun.agreed is None  # a box nobody said yes on builds its boards exactly as before
    plans = AgreedPlans("me")
    plans.on_agreed(EventAgreed(ts=at(9), circle_id="team", thread_id="t1", attendees=["me", "dom"],
                                event={"title": "Team sync", "start": at(19), "end": at(19, 30), "location": "Library", "created_by": "dom"}))
    b = accepted(places, tmp_path / "joined", agreed=plans)
    assert b.dayrun.agreed is b.optional and b.optional.inner is plans
    ids = [x.id for x in b.dayrun.board(at(16, 3)).blocks]
    assert "circle:t1" in ids and DINNER_BLOCK in ids


# ---------- the block ----------


def test_the_block_carries_the_title_the_time_and_the_place_and_never_a_name(places, tmp_path):  # noqa: F811
    b = accepted(places, tmp_path)
    known = b.loop.memory.items(at(0), at(0, 0, 14))
    [item] = b.optional.items(at(0), at(0, 0, 14), TZ, known)
    assert item.model_dump(exclude_none=True) == {"connector": "day", "kind": "calendar_event", "external_id": DINNER_BLOCK,
                                                  "title": "Friends' dinner", "when": at(17, 15), "summary": "until 6:15 PM · '63 South"}
    assert b.optional.items(at(18, 15), at(0, 0, 14), TZ, known) == [] and b.optional.items(at(0), at(17, 15), TZ, known) == []
    # every plan, re-plan, card, status and narration this box published: no friend is named anywhere but in the list
    published = "\n".join(e.model_dump_json() for e in b.out)
    assert names_in(published) == [] and "with Riley" not in published
    assert not [m for m in b.plan().moves if names_in(m.title + m.id + (m.place_name or ""))]
    assert names_in(json.dumps(b.optional.view(at(16, 3)))) == list(FRIENDS)  # the owner's own list is where they are
    # the answer is box-only, and it says which event, never who
    [answer] = b.of(DayOptionalAnswered)
    assert answer.type in __import__("contracts.day", fromlist=["BOX_ONLY_EVENTS"]).BOX_ONLY_EVENTS and names_in(answer.model_dump_json()) == []


def test_a_calendar_item_that_already_covers_it_is_not_there_twice(places, tmp_path):  # noqa: F811
    """A box whose calendar connector wrote the card's event gets it back as an ordinary calendar item."""
    b = accepted(places, tmp_path)
    synced = Item(connector="outlook", kind="calendar_event", external_id="outlook:dinner", title="Friends' dinner", when=at(17, 15),
                  summary="until 6:15 PM · '63 South")
    b.loop.memory._items.append(synced)
    blocks = [x.id for x in b.dayrun.board(at(16, 3)).blocks if x.title == "Friends' dinner"]
    assert blocks == ["outlook:dinner"]


def test_every_seeded_event_lands_on_the_board_at_its_own_place(places, tmp_path):  # noqa: F811
    b = box(places, tmp_path)
    known = b.loop.memory.items(at(0, 0, 11), at(0, 0, 15))
    offers = [o for i in known if (o := rule.parse(i, b.loop.tracker.walks.locate, b.loop.tracker.walks.name, TZ))]
    assert len(offers) == 7
    with b.optional.board_with(add={o.id: None for o in offers}):
        items = b.optional.items(at(0, 0, 11), at(0, 0, 15), TZ, known)
    assert [i.external_id for i in items] == [rule.block_id(o.id) for o in sorted(offers, key=lambda o: (o.start, o.id))]
    for i, o in zip(items, sorted(offers, key=lambda o: (o.start, o.id))):
        where = " · ".join(part for part in i.summary.split("·") if "until" not in part)
        assert b.loop.tracker.walks.locate(where) == o.place and o.place, i.external_id  # the board's own lookup finds the same place
        assert names_in(i.model_dump_json()) == []
    assert b.optional.items(at(0, 0, 11), at(0, 0, 15), TZ, known) == []  # only while the re-plan is being built


def test_the_card_writes_the_event_once_and_later_plans_do_not_write_it_again(places, tmp_path):  # noqa: F811
    b = afternoon(places, tmp_path, writes=True)
    body = said_yes(b)
    [card] = b.approvals.pending()
    assert card.kind == "connector_write" and card.target == "calendar.create_event"
    assert card.reason.startswith("You said yes to Friends' dinner at 5:15 PM. Replace today's ")
    assert "Friends' dinner 5:15 PM" in card.reason and names_in(card.reason + json.dumps(card.params)) == []
    events = card.params["events"]
    assert events[0] == {"title": "Friends' dinner", "start": at(17, 15).isoformat(), "end": at(18, 15).isoformat(), "location": "'63 South"}
    assert all(": " in e["title"] and "dinner" not in e["title"].lower() for e in events[1:]) and len(events) > 1  # the chosen blocks, prefixed
    b.approve(body["action_id"], at(16, 2))
    b.clock["now"] = at(16, 20)
    later, later_card = b.loop.planner.replan(at(16, 20), "apartment.desk", "You asked.", asked=True)
    assert DINNER_BLOCK in [m.id for m in later.moves] and later.for_optional is None
    assert not [e for e in later_card.params["events"] if "dinner" in e["title"].lower()] and "Friends' dinner" not in later_card.reason


# ---------- tracking, the walk, leave now, the review ----------


def test_an_accepted_event_is_tracked_like_any_fixed_event(places, tmp_path):  # noqa: F811
    b = accepted(places, tmp_path)
    b.tick(at(16, 3))
    for minute in range(40, 60, 2):  # still at the desk as the time to leave comes
        b.publish(zone(at(16, minute), "desk", "dwell"))
        b.tick(at(16, minute, ) + timedelta(seconds=30))
    risks = [(s.ts.astimezone(TZ).strftime("%H:%M"), s.risk) for s in b.of(DayStatusChanged) if s.next_event and s.next_event.id == DINNER_BLOCK]
    assert risks[0] == ("16:02", "ok") and risks[-1][1] == "leave_now"  # from the approval on: on time, then time to go
    leave = next(s for s in b.of(DayStatusChanged) if s.risk == "leave_now")
    assert leave.next_event.walk_minutes == 16 and leave.next_event.leave_by == at(16, 59) and at(16, 49) <= leave.ts <= at(16, 59)
    urgent = [n for n in b.of(DayNarrated) if n.moment == "now" and n.urgent]
    assert urgent and "Friends' dinner" in urgent[0].display and names_in(urgent[0].spoken + urgent[0].display) == []
    # the evening review is of the plan that carries it
    b.tick(at(20, 1))
    [review] = b.of(DayReviewed)
    assert review.plan_id == b.plan().plan_id and names_in(review.model_dump_json()) == []


def test_the_yes_is_said_as_the_day_with_the_event_worked_in(places, tmp_path):  # noqa: F811
    b = afternoon(places, tmp_path)
    said_yes(b)
    told = [n for n in b.of(DayNarrated) if n.moment == "replan"][-1]
    assert told.spoken == ("Okay, I've worked Friends' dinner in: first up is Friends' dinner at quarter past five. Leave by five to five for that; "
                           "it's a sixteen-minute walk. Want me to switch to that?")
    assert told.display.startswith("With Friends' dinner added. First up is Friends' dinner at 5:15 PM ('63 South). 16 min walk: leave by 4:59 PM.")
    assert b.loop.narrator.latest["replan"] == told and names_in(told.spoken + told.display) == []


# ---------- what may move, and what never does ----------


def test_a_nearly_fit_moves_only_blocks_atlas_chose(places, tmp_path):  # noqa: F811
    b = afternoon(places, tmp_path)
    before = {m.id: (m.start, m.end) for m in b.plan().moves if m.fixed}
    body = said_yes(b, REVIEW)
    replan = b.loop.planner.pending[MONDAY]
    assert body["fit"]["fits"] == "nearly" and {"Eat at Kitchen", "Study at Desk"} <= {m["title"] for m in body["moves"]}
    assert body["block"]["start"] == at(18, 30).isoformat() and body["block"]["end"] == at(20).isoformat() and body["block"]["kind"] == "study"
    after = {m.id: (m.start, m.end) for m in replan.moves if m.fixed}
    assert all(after[k] == v for k, v in before.items() if k in after and v[1] > at(16, 1))  # no fixed block moved
    assert not [m for m in replan.moves if not m.fixed and m.start < at(20) and m.end > at(18, 30)]  # nothing chosen sits on top of it


def test_leaving_early_for_the_next_fixed_event_is_part_of_the_yes(places, tmp_path):  # noqa: F811
    coffee = offer("coffee", at(9, 25), "10:35 AM", "'63 South")  # ten minutes from the 10:40 lecture: five minutes early
    b = box(places, tmp_path, items=calendar() + [coffee])
    b.morning()
    row = b.row("coffee", at(8, 30))
    assert (row["fit"]["fits"], row["fit"]["leave_early_minutes"]) == ("nearly", 5)
    body = said_yes(b, "coffee", at(8, 30))
    replan = b.loop.planner.pending[MONDAY]
    assert body["block"]["end"] == at(10, 30).isoformat() and b.of(DayOptionalAnswered)[0].until == at(10, 30)
    lecture = next(m for m in replan.moves if m.id == LECTURE)
    assert (lecture.start, lecture.end) == (at(10, 40), at(11, 55))  # on time, and whole
    b.approve(body["action_id"], at(8, 31))
    assert next(m for m in b.plan().moves if m.origin == "optional").end == at(10, 30)
    block = next(x for x in b.dayrun.board(at(8, 32)).blocks if x.id.startswith("optional:"))
    assert block.end == at(10, 30)  # every later plan keeps the early end
    again = box(places, tmp_path, name="b", items=calendar() + [coffee])
    asyncio.run(again.bus.replay_into())
    assert again.optional.until == b.optional.until == {coffee.external_id: at(10, 30)}
    assert next(x for x in again.dayrun.board(at(8, 32)).blocks if x.id.startswith("optional:")).end == at(10, 30)


def test_a_new_yes_is_planned_with_the_events_own_end_not_an_older_early_one(places, tmp_path):  # noqa: F811
    """An earlier yes meant leaving ten minutes early and its card ran out. Asked again when nothing is in the way any
    more, the block is the whole event: the early end belonged to that earlier answer."""
    b = afternoon(places, tmp_path)
    known = b.loop.memory.items(at(0), at(0, 0, 14))
    b.optional.until[DINNER] = at(18, 5)
    with b.optional.board_with(add={DINNER: None}):
        [whole] = b.optional.items(at(0), at(0, 0, 14), TZ, known)
    with b.optional.board_with(add={DINNER: at(18)}):
        [early] = b.optional.items(at(0), at(0, 0, 14), TZ, known)
    assert (whole.summary, early.summary) == ("until 6:15 PM · '63 South", "until 6:00 PM · '63 South")
    body = said_yes(b)
    assert body["block"]["end"] == at(18, 15).isoformat() and DINNER not in b.optional.until  # the new answer has no early end


def test_a_replan_that_would_touch_a_fixed_event_is_refused(places, tmp_path):  # noqa: F811
    """The fit is read from the plan; the re-plan is built from the calendar as it is now. An event the calendar gained
    since the plan was made is caught by comparing the day re-planned with and without the optional event."""
    b = afternoon(places, tmp_path)
    b.loop.memory._items.append(Item(connector="calendar", kind="calendar_event", external_id="outlook:advisor", title="Advisor meeting",
                                     when=at(18), summary="until 6:45 PM · Classroom Building I"))
    with pytest.raises(Refused) as refused:
        b.optional.answer(at(16, 1), DINNER, "yes")
    assert refused.value.reason == "Friends' dinner does not fit: it would cut into Advisor meeting, which is fixed" and refused.value.status == 409
    assert b.out[-1].type != "day.optional.answered" and b.optional._try == {} and b.optional._drop == set()


def test_two_yeses_in_a_row_lose_neither(places, tmp_path):  # noqa: F811
    soccer = offer("frisbee", at(19, 0), "7:45 PM", "Memory Mall", tags="with Sam · free · optional · via GroupMe")
    b = afternoon(places, tmp_path, items=calendar() + near() + [soccer])
    first = said_yes(b, DINNER, at(16, 1))
    second = said_yes(b, "frisbee", at(16, 2))
    replan = b.loop.planner.pending[MONDAY]
    carried = sorted(m.id for m in replan.moves if m.origin == "optional")
    assert replan.plan_id == second["plan_id"] != first["plan_id"] and carried == [DINNER_BLOCK, rule.block_id(soccer.external_id)]
    for key in (DINNER, "frisbee"):  # both wait on the newer card
        row = b.row(key, at(16, 2))
        assert (row["awaiting_approval"], row["action_id"]) == (True, second["action_id"])
    b.approve(second["action_id"], at(16, 3))
    assert b.optional.accepted == {DINNER, soccer.external_id}
    # and one that would sit on top of the first is refused while the first waits
    b2 = afternoon(places, tmp_path / "two", items=calendar() + near() + [offer("tacos", at(17, 30), "6:30 PM", "Student Union")])
    said_yes(b2, DINNER, at(16, 1))
    with pytest.raises(Refused, match="Tacos does not fit: it overlaps Friends' dinner, which is fixed"):
        b2.optional.answer(at(16, 2), "tacos", "yes")


# ---------- a no ----------


def test_a_no_is_recorded_and_a_change_of_mind_is_the_newest_answer(places, tmp_path):  # noqa: F811
    b = afternoon(places, tmp_path)
    events, body = b.optional.answer(at(16, 1), DINNER, "no")
    assert [e.type for e in events] == ["day.optional.answered"] and body == {"id": DINNER, "answer": "no", "state": "declined", "proposed": False}
    b.publish(*events)
    assert b.row(DINNER, at(16, 1))["state"] == "declined" and MONDAY not in b.loop.planner.pending
    yes = said_yes(b, DINNER, at(16, 5))
    assert yes["proposed"] and b.row(DINNER, at(16, 5))["answer"] == "yes" and b.row(DINNER, at(16, 5))["state"] == "suggested"
    again = box(places, tmp_path, name="b")
    asyncio.run(again.bus.replay_into())
    assert again.optional.answers[DINNER].answer == "yes" and again.optional.view(at(16, 5)) == b.optional.view(at(16, 5))


def test_a_no_to_an_accepted_event_takes_it_out_through_a_card_too(places, tmp_path):  # noqa: F811
    b = accepted(places, tmp_path)
    with_dinner = b.plan()
    b.clock["now"] = at(16, 10)
    events, body = b.optional.answer(at(16, 10), DINNER, "no")
    b.publish(*events)
    replan = b.loop.planner.pending[MONDAY]
    assert (body["state"], body["proposed"], body["plan_id"], body["action_id"]) == ("accepted", True, replan.plan_id, replan.action_id)
    assert replan.for_optional == DINNER and DINNER_BLOCK not in [m.id for m in replan.moves]
    assert b.plan() == with_dinner and b.optional.accepted == {DINNER}  # still in the plan until the card is approved
    row = b.row(DINNER, at(16, 10))
    assert (row["state"], row["answer"], row["awaiting_approval"], row["action_id"]) == ("accepted", "no", True, replan.action_id)
    told = [n for n in b.of(DayNarrated) if n.moment == "replan"][-1]
    assert told.spoken.startswith("Okay, Friends' dinner is out: ") and told.display.startswith("Without Friends' dinner.")
    card = next(r for r in b.approvals.pending() if r.id == replan.action_id)
    assert card.reason.startswith("You took Friends' dinner out of today. Replace today's ")
    b.approve(replan.action_id, at(16, 11))
    assert b.optional.accepted == set() and b.plan().plan_id == replan.plan_id and b.row(DINNER, at(16, 11))["state"] == "declined"
    assert DINNER_BLOCK not in [x.id for x in b.dayrun.board(at(16, 12)).blocks]
    again = box(places, tmp_path, name="b")
    asyncio.run(again.bus.replay_into())
    assert again.optional.accepted == set() and again.optional.view(at(16, 12)) == b.optional.view(at(16, 12))


def test_a_yes_to_one_already_in_the_plan_is_refused(places, tmp_path):  # noqa: F811
    b = accepted(places, tmp_path)
    with pytest.raises(Refused, match="Friends' dinner is already in your plan"):
        b.optional.answer(at(16, 5), DINNER, "yes")
    assert b.row(DINNER, at(18, 20))["state"] == "passed"  # and once it is over it is over


# ---------- the leaving checklist ----------


def test_the_leaving_checklist_counts_an_optional_event_only_once_it_is_accepted(places, tmp_path):  # noqa: F811
    from contexts.agent.application.planner import Planner
    from contexts.agent.application.qa import QA
    from runtime.fake import FakeRuntime

    b = afternoon(places, tmp_path)
    qa = QA(FakeRuntime(), b.loop.memory, clock=lambda: at(16, 30))
    planner = Planner(qa)
    assert planner.accepted_optional() == ()
    assert "dinner" not in planner.leaving(at(16, 30)).text  # on offer: not what you are leaving for
    planner.accepted_optional = lambda: b.optional.accepted  # the composition root's line (contexts/gateway/bootstrap.py)
    assert "dinner" not in planner.leaving(at(16, 30)).text
    body = said_yes(b)
    assert "dinner" not in planner.leaving(at(16, 30)).text  # said yes, card not approved: still not a commitment
    b.approve(body["action_id"], at(16, 2))
    text = planner.leaving(at(16, 30)).text
    assert text.startswith("Next: Friends' dinner at 5:15 PM; leave by ") and "COP4533 review session" not in text and names_in(text) == []
