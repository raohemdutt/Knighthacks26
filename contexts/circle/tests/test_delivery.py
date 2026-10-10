import asyncio
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from contexts.circle.application.delivery import AgreedEventDelivery
from contexts.connectors.application.hub import ConnectorHub
from contexts.connectors.domain.spec import ConnectorSpec
from contexts.connectors.fakes.fake import FakeConnector
from contexts.workspace.domain.approvals import Approvals
from contracts.circle import EventAgreed, Proposal
from contracts.experience import ApprovalDecided

NY = ZoneInfo("America/New_York")
START = datetime(2026, 10, 11, 14, tzinfo=NY)
AGREED = EventAgreed(ts=START - timedelta(hours=20), circle_id="team", thread_id="t1",
                     event=Proposal(title="Study session", start=START, end=START + timedelta(hours=2), location="Library",
                                    invitees=["hemdutt"]),
                     attendees=["dom", "hemdutt"])


def hub(*names):
    specs = {"calendar": ConnectorSpec(name="calendar", module="x", writes=["create_event"]),
             "groupme": ConnectorSpec(name="groupme", module="x", writes=["post"])}
    fakes = {n: FakeConnector() for n in names}
    return ConnectorHub([specs[n] for n in names], factory=lambda s: fakes[s.name]), fakes


def test_agreement_proposes_invite_and_group_message_behind_approvals():
    h, fakes = hub("calendar", "groupme")
    d = AgreedEventDelivery(h, "dom", NY, emails={"hemdutt": "hemdutt@example.edu"}, public_url="https://atlas.ts.net")
    invite, chat = d.on_agreed(AGREED)
    assert (invite.target, invite.params["attendees"], invite.params["location"]) == (
        "calendar.create_event", ["hemdutt@example.edu"], "Library")
    assert chat.target == "groupme.post" and chat.params["group"] == "friends"
    assert chat.params["text"] == "Study session: Sun Oct 11, 2:00 PM at Library. Everyone's in. Invites: https://atlas.ts.net/api/invites"

    approvals = Approvals({"connector_write": h.execute_write})
    for p in (invite, chat):
        approvals.on_proposed(p)
    assert fakes["calendar"].calls == [] and fakes["groupme"].calls == []  # nothing until approved
    asyncio.run(approvals.on_decided(ApprovalDecided(ts=AGREED.ts, action_id=invite.action_id, decision="approve")))
    assert fakes["calendar"].calls[0][0] == "create_event" and fakes["groupme"].calls == []


def test_only_declared_writes_are_proposed():
    h, _ = hub("calendar")
    [only] = AgreedEventDelivery(h, "dom", NY).on_agreed(AGREED)
    assert only.target == "calendar.create_event"
