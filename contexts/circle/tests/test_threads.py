"""The agents' conversation: threads rebuilt from the log, enriched from the store, and their plain lines."""

import asyncio
from datetime import datetime, timedelta
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from fastapi import FastAPI
from fastapi.testclient import TestClient

from bus.bus import EventBus
from contexts.circle.api import circle_router
from contexts.circle.application.protocol import AgentMessage, InMemoryCircleStore
from contexts.circle.application.sharing import CircleOffline
from contexts.circle.application.threads import Members, Threads, agreed_line, proposal_line, reply_line
from contracts.agent import ActionProposed
from contracts.circle import EventAgreed, MessageReceived, PlaceShared, Proposal, Reply
from contracts.play import LeagueSummaryReceived
from contracts.workspace import ActionCompleted, ApprovalRequested

TZ = ZoneInfo("America/New_York")
NOW = datetime(2026, 10, 10, 15, 0, tzinfo=TZ)  # Saturday 3:00 PM
NAMES = {"hemdutt": "Hemdutt", "maya": "Maya", "dominick": "Dominick"}
MEMBERS = list(NAMES)
T0 = NOW - timedelta(hours=1)


def at(day: int, hour: int, minute: int = 0) -> datetime:
    return datetime(2026, 10, day, hour, minute, tzinfo=TZ)


STUDY = Proposal(title="COP4533 study session", start=at(11, 14), end=at(11, 16), location="the Library", invitees=["dominick", "maya"])
LATER = STUDY.model_copy(update={"start": at(11, 15, 30), "end": at(11, 17, 30)})


def s(n: int) -> timedelta:
    return timedelta(seconds=n)


def card(thread: str, p: Proposal, ts: datetime, sent: datetime | None = None) -> list:
    """This box's proposal: the approval card, and (when approved) the completion that sent it."""
    out = [ActionProposed(event_id=f"{thread}-card", ts=ts, action_id=f"a-{thread}", kind="circle_share", target="team", reason="ask",
                          params={"kind": "event_proposal", "thread_id": thread, "proposal": p.model_dump(mode="json")}),
           ApprovalRequested(event_id=f"{thread}-req", ts=ts, action_id=f"a-{thread}", kind="circle_share", target="team", reason="ask",
                             expires_at=ts + timedelta(minutes=5))]
    if sent:
        out.append(ActionCompleted(event_id=f"{thread}-done", ts=sent, action_id=f"a-{thread}", status="done", result="proposed"))
    return out


def received(thread: str, member: str, ts: datetime, reply: Reply | None = None, proposal: Proposal | None = None) -> MessageReceived:
    return MessageReceived(event_id=f"{thread}-{member}-ev", ts=ts, message_id=f"{thread}-{member}", circle_id="team", thread_id=thread,
                           from_member=member, kind="proposal" if proposal else "reply", proposal=proposal, reply=reply)


def agreed(thread: str, p: Proposal, ts: datetime, by: str = "hemdutt") -> EventAgreed:
    return EventAgreed(event_id=f"{thread}-agreed", ts=ts, circle_id="team", thread_id=thread, event=p, attendees=sorted(NAMES),
                       created_by=by)


def counter_then_agreed() -> list:
    """Hemdutt asks; Dominick is free, Maya counters; the new time goes out as a second thread and is agreed."""
    return [
        *card("t1", STUDY, T0, sent=T0 + s(10)),
        received("t1", "dominick", T0 + s(38), Reply(answer="free")),
        received("t1", "maya", T0 + s(40), Reply(answer="counter", counter=LATER)),
        *card("t2", LATER, T0 + s(40), sent=T0 + s(70)),
        received("t2", "dominick", T0 + s(98), Reply(answer="free")),
        received("t2", "maya", T0 + s(127), Reply(answer="free")),
        agreed("t2", LATER, T0 + s(127)),
    ]


def project(events: list) -> Threads:
    threads = Threads()
    handlers = {"agent.action.proposed": threads.on_proposed, "workspace.approval.requested": threads.on_requested,
                "workspace.action.completed": threads.on_completed, "circle.message.received": threads.on_message,
                "circle.event.agreed": threads.on_agreed}
    for e in events:
        handlers[e.type](e)
    return threads


def view(events: list, now: datetime = NOW, extra: list | None = None, me: str = "hemdutt") -> list[dict]:
    return project(events).view("team", me, NAMES, MEMBERS, now, TZ, extra=extra)


def lines(thread: dict) -> list[str]:
    return [m["line"] for m in thread["messages"]]


# --- the plain lines ---


def test_a_proposal_reads_as_a_question_from_that_members_atlas():
    assert proposal_line("Hemdutt", STUDY, ["Dominick", "Maya"], True, NOW, TZ) == \
        "Hemdutt's Atlas: Can everyone do COP4533 study session, tomorrow 2:00 to 4:00 PM at the Library?"
    one = STUDY.model_copy(update={"start": at(10, 11), "end": at(10, 13), "location": None, "invitees": ["maya"]})
    assert proposal_line("Hemdutt", one, ["Maya"], False, NOW, TZ) == \
        "Hemdutt's Atlas: Can Maya do COP4533 study session, today 11:00 AM to 1:00 PM?"
    far = STUDY.model_copy(update={"start": at(14, 9), "end": at(14, 10)})
    assert "Wed Oct 14 9:00 to 10:00 AM" in proposal_line("Hemdutt", far, [], True, NOW, TZ)


def test_a_reply_line_never_says_more_than_the_reply_carries():
    """A reply is one word and, for a counter, a time. There is no reason in it, so no line gives one."""
    free = reply_line("Dominick", Reply(answer="free"), STUDY, NOW, TZ)
    busy = reply_line("Maya", Reply(answer="busy"), STUDY, NOW, TZ)
    counter = reply_line("Maya", Reply(answer="counter", counter=LATER), STUDY, NOW, TZ)
    other_day = reply_line("Maya", Reply(answer="counter", counter=STUDY.model_copy(update={"start": at(12, 9), "end": at(12, 11)})), STUDY, NOW, TZ)
    assert free == "Dominick's Atlas: Free."
    assert busy == "Maya's Atlas: Busy then."
    assert counter == "Maya's Atlas: Busy then. Suggests 3:30 PM."
    assert other_day == "Maya's Atlas: Busy then. Suggests Mon Oct 12 9:00 AM."
    assert reply_line("Maya", Reply(answer="counter"), STUDY, NOW, TZ) == "Maya's Atlas: Busy then."  # a counter with no time
    for line in (free, busy, counter, other_day):
        for invented in ("lecture", "class", "because", "until", "meeting", "calendar", "sorry"):
            assert invented not in line.lower()


def test_the_agreement_line_counts_who_is_in():
    assert agreed_line(agreed("t2", LATER, NOW), NOW, TZ) == "Agreed: tomorrow 3:30 PM, the Library, 3 of 3 in."


def test_an_older_message_names_the_day_instead_of_saying_tomorrow():
    """Read on Saturday, "tomorrow" inside a message sent on Friday would point at the wrong day."""
    friday = NOW - timedelta(days=1)
    assert "Sun Oct 11 2:00 to 4:00 PM" in proposal_line("Hemdutt", STUDY, [], True, NOW, TZ, said=friday)
    assert "tomorrow 2:00 to 4:00 PM" in proposal_line("Hemdutt", STUDY, [], True, NOW, TZ, said=NOW - timedelta(hours=3))
    assert agreed_line(agreed("t2", LATER, friday), NOW, TZ) == "Agreed: Sun Oct 11 3:30 PM, the Library, 3 of 3 in."


# --- the projection ---


def test_counter_then_agreed_is_one_conversation_in_order():
    [t] = view(counter_then_agreed())
    assert (t["thread_id"], t["legs"], t["state"], t["mine"]) == ("t1", ["t1", "t2"], "agreed", True)
    assert lines(t) == [
        "Hemdutt's Atlas: Can everyone do COP4533 study session, tomorrow 2:00 to 4:00 PM at the Library?",
        "Dominick's Atlas: Free.",
        "Maya's Atlas: Busy then. Suggests 3:30 PM.",
        "Hemdutt's Atlas: Can everyone do tomorrow 3:30 to 5:30 PM instead?",
        "Dominick's Atlas: Free.",
        "Maya's Atlas: Free.",
        "Agreed: tomorrow 3:30 PM, the Library, 3 of 3 in.",
    ]
    assert [m["at"] for m in t["messages"]] == sorted(m["at"] for m in t["messages"])
    assert [m["kind"] for m in t["messages"]] == ["proposal", "reply", "reply", "proposal", "reply", "reply", "agreed"]
    assert (t["starts_at"], t["location"]) == (LATER.start.isoformat(), "the Library")  # the time that was agreed
    assert [(i["display_name"], i["answer"]) for i in t["invitees"]] == [("Dominick", "free"), ("Maya", "free")]
    assert t["waiting_on"] == [] and t["action"]["kind"] == "none"


def test_the_states_a_thread_moves_through():
    events = counter_then_agreed()

    def state(n: int, now: datetime = NOW) -> tuple:
        found = view(events[:n], now)
        return (found[0]["state"], found[0]["action"]["kind"], [w["display_name"] for w in found[0]["waiting_on"]]) if found else None

    assert state(2, T0 + s(5)) == ("proposed", "approve", [])  # the card is up; nothing sent
    assert state(3) == ("waiting", "none", ["Dominick", "Maya"])
    assert state(4) == ("waiting", "none", ["Maya"])
    assert state(5) == ("countered", "propose", [])  # Maya countered; no card for the new time in the log yet
    assert state(7, T0 + s(45)) == ("countered", "approve", [])  # the follow-up card is waiting
    assert state(7, NOW) == ("countered", "propose", [])  # ... and ran out unanswered: offer to ask again
    assert state(8) == ("waiting", "none", ["Dominick", "Maya"])  # the new time is out
    assert state(10) == ("waiting", "none", [])  # everyone free, the agreement not published yet
    assert state(11) == ("agreed", "none", [])

    pending = view(events[:7], T0 + s(45))[0]
    assert pending["pending_action_id"] == "a-t2" and pending["action"]["action_id"] == "a-t2"
    assert pending["messages"][-1]["kind"] == "note" and pending["messages"][-1]["line"].startswith("Not sent yet.")
    again = view(events[:5])[0]["action"]["proposal"]
    assert (again["start"], again["invitees"]) == (LATER.start.isoformat(), ["dominick", "maya"])


def test_a_card_that_was_never_approved_is_not_a_conversation():
    assert view(card("t1", STUDY, T0), NOW) == []  # ran out
    denied = [*card("t1", STUDY, T0), ActionCompleted(ts=T0 + s(5), action_id="a-t1", status="denied")]
    assert view(denied, T0 + s(6)) == []
    other = ActionProposed(ts=T0, action_id="x", kind="circle_share", target="team", reason="stake", params={"kind": "league_stake"})
    assert view([other], T0) == []  # the League's own shares are not plans


def test_busy_with_no_other_time_is_declined():
    events = [*card("t1", STUDY, T0, sent=T0 + s(10)), received("t1", "maya", T0 + s(40), Reply(answer="busy"))]
    [t] = view(events)
    assert (t["state"], t["action"]["kind"]) == ("declined", "propose")
    assert lines(t)[-1] == "Maya's Atlas: Busy then."


def test_a_friends_plan_shows_this_member_as_free_only_once_it_is_agreed():
    """This box's own reply is not in its log. An agreement means every attendee said free, so it shows then."""
    dinner = Proposal(title="Dinner", start=at(10, 18, 30), end=at(10, 19, 30), location="the Student Union")
    events = [received("d", "maya", T0, proposal=dinner), received("d", "dominick", T0 + s(30), Reply(answer="free"))]
    [open_] = view(events)
    assert (open_["state"], open_["mine"], [w["display_name"] for w in open_["waiting_on"]]) == ("waiting", False, ["Hemdutt"])
    assert lines(open_) == ["Maya's Atlas: Can everyone do Dinner, today 6:30 to 7:30 PM at the Student Union?", "Dominick's Atlas: Free."]

    [done] = view([*events, agreed("d", dinner, T0 + s(60), by="maya")])
    assert lines(done)[2:] == ["Hemdutt's Atlas: Free.", "Agreed: today 6:30 PM, the Student Union, 3 of 3 in."]
    assert done["messages"][2]["source"] == "agreement" and done["state"] == "agreed"


def test_the_store_adds_what_the_log_lacks_and_never_duplicates():
    dinner = Proposal(title="Dinner", start=at(10, 18, 30), end=at(10, 19, 30))
    events = [received("d", "maya", T0, proposal=dinner), *card("t1", STUDY, T0, sent=T0 + s(10))]
    rows = [
        AgentMessage(id="d-maya", circle_id="team", thread_id="d", from_member="maya", kind="proposal", proposal=dinner, created_at=T0),  # logged
        AgentMessage(id="mine", circle_id="team", thread_id="d", from_member="hemdutt", kind="reply", reply=Reply(answer="busy"), created_at=T0 + s(20)),
        AgentMessage(id="row-1", circle_id="team", thread_id="t1", from_member="hemdutt", kind="proposal", proposal=STUDY, created_at=T0 + s(10)),  # my card
        AgentMessage(id="lost", circle_id="team", thread_id="old", from_member="dominick", kind="proposal", proposal=dinner, created_at=T0 - s(900)),
        AgentMessage(id="else", circle_id="elsewhere", thread_id="x", from_member="maya", kind="proposal", proposal=dinner, created_at=T0),
    ]
    found = {t["thread_id"]: t for t in view(events, extra=rows)}
    assert set(found) == {"d", "t1", "old"}
    assert lines(found["d"]) == ["Maya's Atlas: Can everyone do Dinner, today 6:30 to 7:30 PM?", "Hemdutt's Atlas: Busy then."]
    assert [m["source"] for m in found["d"]["messages"]] == ["log", "store"] and found["d"]["state"] == "declined"
    assert len(found["t1"]["messages"]) == 1  # the store's row for my own proposal is the card already shown
    assert found["old"]["messages"][0]["source"] == "store"


def test_threads_are_listed_by_latest_activity():
    dinner = Proposal(title="Dinner", start=at(10, 18, 30), end=at(10, 19, 30))
    events = [received("d", "maya", T0 + s(500), proposal=dinner), *counter_then_agreed()]
    assert [t["thread_id"] for t in view(events)] == ["d", "t1"]


# --- the endpoints ---


def app(tmp_path, **kw):
    (tmp_path / "agent.yaml").write_text("sleep:\n  timezone: America/New_York\ncircle:\n  names: {hemdutt: Hemdutt, maya: Maya, dominick: Dominick}\n")
    bus = EventBus(tmp_path / "events")
    api = FastAPI()
    api.include_router(circle_router(bus, clock=lambda: NOW, config_dir=tmp_path, **kw))
    return bus, TestClient(api)


def me(store=None):
    return SimpleNamespace(circle_id="team", member_id="hemdutt", members=MEMBERS, store=store)


def test_without_a_circle_the_threads_and_members_say_so(tmp_path):
    bus, c = app(tmp_path)
    asyncio.run(bus.publish_all(counter_then_agreed()))  # even with a story in the log
    assert c.get("/api/circle/threads").json() == {"configured": False, "circle_id": None, "member_id": None, "source": "none",
                                                   "offline": False, "members": [], "threads": []}
    assert c.get("/api/circle/threads/t1").status_code == 404
    members = c.get("/api/circle/members").json()
    assert members["configured"] is False and members["members"] == [] and len(members["connect"]) == 2


def test_the_threads_endpoints_read_the_log(tmp_path):
    bus, c = app(tmp_path, circle=me())
    asyncio.run(bus.publish_all(counter_then_agreed()))
    body = c.get("/api/circle/threads").json()
    assert (body["configured"], body["source"], body["offline"]) == (True, "log", False)
    assert [m["display_name"] for m in body["members"]] == ["Hemdutt", "Maya", "Dominick"] and body["members"][0]["mine"]
    [t] = body["threads"]
    assert "messages" not in t and t["messages_count"] == 7 and t["last_line"].startswith("Agreed: tomorrow 3:30 PM")
    for leg in ("t1", "t2"):  # either id finds the conversation
        one = c.get(f"/api/circle/threads/{leg}").json()
        assert one["thread_id"] == "t1" and len(one["messages"]) == 7
    assert c.get("/api/circle/threads/nope").status_code == 404


def test_the_threads_are_the_same_after_a_restart(tmp_path):
    bus, c = app(tmp_path, circle=me())
    asyncio.run(bus.publish_all(counter_then_agreed()))
    live = c.get("/api/circle/threads/t1").json()
    bus, c = app(tmp_path, circle=me())  # a new process on the same log
    assert c.get("/api/circle/threads").json()["threads"] == []
    asyncio.run(bus.replay_into())
    assert c.get("/api/circle/threads/t1").json() == live


def test_the_endpoint_enriches_from_the_store_and_survives_it_being_down(tmp_path):
    store = InMemoryCircleStore()
    dinner = Proposal(title="Dinner", start=at(10, 18, 30), end=at(10, 19, 30))
    store.post(AgentMessage(id="d-maya", circle_id="team", thread_id="d", from_member="maya", kind="proposal", proposal=dinner, created_at=T0))
    store.post(AgentMessage(id="mine", circle_id="team", thread_id="d", from_member="hemdutt", kind="reply", reply=Reply(answer="free"), created_at=T0 + s(20)))
    bus, c = app(tmp_path, circle=me(store))
    asyncio.run(bus.publish(received("d", "maya", T0, proposal=dinner)))
    body = c.get("/api/circle/threads").json()
    assert body["source"] == "log+store" and body["threads"][0]["invitees"][1] == {
        "member_id": "hemdutt", "display_name": "Hemdutt", "mine": True, "answer": "free"}

    class Down:
        def since(self, circle_id, after):
            raise CircleOffline("Neon is unreachable")

    bus, c = app(tmp_path, circle=me(Down()))
    asyncio.run(bus.replay_into())
    body = c.get("/api/circle/threads").json()
    assert (body["source"], body["offline"], body["threads"][0]["messages_count"]) == ("log", True, 1)

    class KnownDown:
        db = SimpleNamespace(offline="Neon is unreachable")

        def since(self, circle_id, after):
            raise AssertionError("the store was read although the timer already found it down")

    bus, c = app(tmp_path, circle=me(KnownDown()))
    assert c.get("/api/circle/threads").json()["offline"] is True


def test_members_show_places_and_whether_a_box_was_seen(tmp_path):
    bus, c = app(tmp_path, circle=me())
    asyncio.run(bus.publish_all([
        PlaceShared(ts=NOW - s(600), circle_id="team", member_id="maya", place="Library", source="phone", expires_at=NOW + s(3600)),
        PlaceShared(ts=NOW - s(9000), circle_id="team", member_id="dominick", place="Gym", source="phone", expires_at=NOW - s(60)),
        LeagueSummaryReceived(ts=NOW, circle_id="team", member_id="dominick", kind="habit_summary", payload={}),
    ]))
    body = c.get("/api/circle/members").json()
    rows = {m["member_id"]: m for m in body["members"]}
    assert [m["member_id"] for m in body["members"]] == ["hemdutt", "maya", "dominick"]
    assert rows["hemdutt"]["mine"] and rows["hemdutt"]["has_box"] is True
    assert rows["maya"]["place"]["place"] == "Library" and rows["maya"]["has_box"] is None  # only a phone share so far
    assert rows["dominick"]["place"] is None and (rows["dominick"]["has_box"], rows["dominick"]["box_seen"]) == (True, "summary")


def test_members_projection_counts_a_message_as_a_box():
    people = Members()
    people.box_seen.setdefault("maya", "agent")
    [me_, maya] = people.view("team", "hemdutt", NAMES, ["hemdutt", "maya"], NOW)
    assert maya["has_box"] is True and maya["box_seen"] == "agent" and me_["box_seen"] == "this box"
