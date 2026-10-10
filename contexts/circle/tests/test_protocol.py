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

    [received] = hem.poll()  # hemdutt's box reads the proposal and replies on its own
    assert isinstance(received, MessageReceived) and received.kind == "proposal"
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
