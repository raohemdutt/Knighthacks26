from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from contexts.agent.application.memory_port import FakeMemory
from contexts.circle.application.protocol import CircleAgent, InMemoryCircleStore
from contexts.workspace.domain.approvals import ActionRequest
from contracts.circle import EventAgreed, MessageReceived, Proposal
from contracts.memory import Item

NY = ZoneInfo("America/New_York")
NOW = datetime(2026, 10, 10, 12, tzinfo=NY)
SUNDAY_2PM = datetime(2026, 10, 11, 14, tzinfo=NY)


class Clock:
    def __init__(self):
        self.t = NOW

    def __call__(self):
        self.t += timedelta(seconds=1)
        return self.t


def agent(member, store, clock, calendar=()):
    items = [Item(connector="calendar", kind="calendar_event", external_id=f"{member}-{i}", title=t, when=w)
             for i, (t, w) in enumerate(calendar)]
    return CircleAgent(member, "team", store, FakeMemory(items=items), clock)


def approve(a: CircleAgent, proposal: Proposal):
    """What Workspace does after the person taps Approve."""
    p = a.propose(proposal, "Study session")
    req = ActionRequest(id=p.action_id, kind=p.kind, target=p.target, reason=p.reason, params=p.params, expires_at=NOW)
    return a.share_executor(req), p.params["thread_id"]


def test_two_atlases_agree_when_both_are_free():
    store, clock = InMemoryCircleStore(), Clock()
    dom, hem = agent("dom", store, clock), agent("hemdutt", store, clock)
    session = Proposal(title="Study session", start=SUNDAY_2PM, end=SUNDAY_2PM + timedelta(hours=2), invitees=["hemdutt"])
    assert approve(dom, session)[0] == "proposed to the circle: Study session"

    received, heard = hem.poll()  # hemdutt's box reads the proposal, replies on its own, and hears it is on
    assert isinstance(received, MessageReceived) and received.kind == "proposal"
    assert isinstance(heard, EventAgreed) and heard.created_by == "dom"  # only dom's box sends the invites
    reply = [m for m in store.messages if m.kind == "reply"][0]
    assert reply.reply.answer == "free" and reply.from_member == "hemdutt"

    out = dom.poll()
    agreed = [e for e in out if isinstance(e, EventAgreed)]
    assert len(agreed) == 1 and agreed[0].attendees == ["dom", "hemdutt"]
    assert store.events  # recorded in the shared calendar
    assert dom.poll() == []  # nothing new, no second agreement


def test_busy_member_counters_with_the_next_free_slot_and_no_agreement():
    store, clock = InMemoryCircleStore(), Clock()
    dom = agent("dom", store, clock)
    xan = agent("xander", store, clock, calendar=[("Shift at work", SUNDAY_2PM + timedelta(minutes=30))])
    approve(dom, Proposal(title="Study", start=SUNDAY_2PM, end=SUNDAY_2PM + timedelta(hours=1), invitees=["xander"]))
    xan.poll()
    reply = [m for m in store.messages if m.kind == "reply"][0].reply
    assert reply.answer == "counter" and reply.counter.start == SUNDAY_2PM + timedelta(hours=2)
    assert not any(isinstance(e, EventAgreed) for e in dom.poll())


def test_only_invitees_reply_and_nothing_private_leaves_the_box():
    store, clock = InMemoryCircleStore(), Clock()
    dom = agent("dom", store, clock)
    ilan = agent("ilan", store, clock, calendar=[("Therapy appointment", SUNDAY_2PM)])
    approve(dom, Proposal(title="Lunch", start=SUNDAY_2PM, end=SUNDAY_2PM + timedelta(hours=1), invitees=["hemdutt"]))
    ilan.poll()
    assert all(m.kind == "proposal" for m in store.messages)  # not invited: no reply
    approve(dom, Proposal(title="Dinner", start=SUNDAY_2PM, end=SUNDAY_2PM + timedelta(hours=1), invitees=["ilan"]))
    ilan.poll()
    shared = [m.model_dump_json() for m in store.messages if m.from_member == "ilan"]
    assert shared and not any("Therapy" in s for s in shared)


def test_a_restart_never_answers_or_agrees_twice():
    """Q12: _seen and _agreed lived in memory, so a restart replied again and published a second agreement."""
    store, clock = InMemoryCircleStore(), Clock()
    dom, hem = agent("dom", store, clock), agent("hemdutt", store, clock)
    approve(dom, Proposal(title="Study", start=SUNDAY_2PM, end=SUNDAY_2PM + timedelta(hours=1), invitees=["hemdutt"]))
    log = hem.poll()
    [agreed] = [e for e in dom.poll() if isinstance(e, EventAgreed)]
    hem2, dom2 = agent("hemdutt", store, clock), agent("dom", store, clock)  # both boxes restart
    for e in log:  # what the log replays into the new agents
        (hem2.on_message if isinstance(e, MessageReceived) else hem2.on_agreed)(e)
    dom2.on_agreed(agreed)
    for e in [e for e in dom.poll() if isinstance(e, MessageReceived)]:
        dom2.on_message(e)
    assert hem2.poll() == [] and len([m for m in store.messages if m.kind == "reply"]) == 1
    assert not any(isinstance(e, EventAgreed) for e in dom2.poll())
    lost = agent("hemdutt", store, clock)  # even with the log gone, the reply already in the thread is not repeated
    lost.poll()
    assert len([m for m in store.messages if m.kind == "reply"]) == 1


def test_a_reply_stamped_by_a_slow_clock_is_still_read():
    """Q12: the invitee's clock ran 60 s behind, its reply sorted before what dom had already seen, and no agreement."""
    store, clock = InMemoryCircleStore(), Clock()
    dom, hem = agent("dom", store, clock), agent("hemdutt", store, clock)
    approve(dom, Proposal(title="Study", start=SUNDAY_2PM, end=SUNDAY_2PM + timedelta(hours=1), invitees=["hemdutt"]))
    dom.poll()  # dom has seen its own proposal
    hem.clock = lambda: clock.t - timedelta(seconds=60)
    hem.poll()
    assert any(isinstance(e, EventAgreed) for e in dom.poll())


def test_a_counter_offers_the_new_time_and_no_invitees_means_the_whole_circle():
    store, clock = InMemoryCircleStore(), Clock()
    dom = agent("dom", store, clock)
    xan = agent("xander", store, clock, calendar=[("Shift at work", SUNDAY_2PM)])
    approve(dom, Proposal(title="Study", start=SUNDAY_2PM, end=SUNDAY_2PM + timedelta(hours=1), invitees=["xander"]))
    xan.poll()
    [card] = [e for e in dom.poll() if getattr(e, "kind", None) == "circle_share"]
    assert card.params["proposal"]["invitees"] == ["xander"] and "suggested" in card.reason
    assert card.params["proposal"]["start"].startswith("2026-10-11T15:00")

    store, clock = InMemoryCircleStore(), Clock()
    team = ["dom", "hemdutt", "ilan"]
    dom, hem, ilan = (CircleAgent(m, "team", store, FakeMemory(), clock, members=team) for m in team)
    approve(dom, Proposal(title="Dinner", start=SUNDAY_2PM, end=SUNDAY_2PM + timedelta(hours=1)))
    hem.poll()
    assert not any(isinstance(e, EventAgreed) for e in dom.poll())  # ilan has not answered yet
    ilan.poll()
    [agreed] = [e for e in dom.poll() if isinstance(e, EventAgreed)]
    assert agreed.attendees == team and agreed.event.invitees == ["hemdutt", "ilan"]
