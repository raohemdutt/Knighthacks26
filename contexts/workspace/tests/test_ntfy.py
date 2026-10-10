import asyncio
import json
from datetime import UTC, datetime

import httpx
import pytest

from contexts.workspace.domain.approvals import Approvals
from contexts.workspace.infrastructure.ntfy import Ntfy, PhoneNotifier, send_message_executor
from contracts.agent import ActionProposed, AlertRaised
from contracts.experience import ApprovalDecided

T0 = datetime(2026, 10, 10, 12, tzinfo=UTC)


def ntfy():
    sent = []

    def handle(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={"id": "x"})

    return Ntfy("http://ntfy.test", transport=httpx.MockTransport(handle)), sent


def test_send_message_runs_only_after_approve_and_only_to_listed_people():
    client, sent = ntfy()
    approvals = Approvals({"send_message": send_message_executor(client, {"xander": "atlas-xander"})})
    for aid, target in (("a1", "xander"), ("a2", "stranger")):
        approvals.on_proposed(ActionProposed(ts=T0, action_id=aid, kind="send_message", target=target, reason="r", params={"text": "Package at the door"}))
    assert sent == []
    [ok] = asyncio.run(approvals.on_decided(ApprovalDecided(ts=T0, action_id="a1", decision="approve")))
    [bad] = asyncio.run(approvals.on_decided(ApprovalDecided(ts=T0, action_id="a2", decision="approve")))
    assert (ok.status, bad.status) == ("done", "failed")
    assert [(m["topic"], m["message"]) for m in sent] == [("atlas-xander", "Package at the door")]


def test_approval_card_has_buttons_that_post_back():
    client, sent = ntfy()
    notifier = PhoneNotifier(client, gateway_url="https://atlas.tailnet.ts.net")
    proposals = Approvals()
    [req] = proposals.on_proposed(ActionProposed(ts=T0, action_id="a9", kind="connector_write", target="calendar.create_event", reason="Study session Sunday 2 PM"))
    asyncio.run(notifier.on_approval(req))
    asyncio.run(notifier.on_alert(AlertRaised(alert_id="x", rule_id="package-at-door", message="Package left at the door")))
    card, alert = sent
    assert [a["label"] for a in card["actions"]] == ["Approve", "Deny"]
    assert card["actions"][0]["url"] == "https://atlas.tailnet.ts.net/api/approvals/a9"
    assert json.loads(card["actions"][1]["body"]) == {"decision": "deny"}
    assert alert["message"] == "Package left at the door" and alert["priority"] == 4


def test_publish_raises_on_server_error():
    client = Ntfy("http://ntfy.test", transport=httpx.MockTransport(lambda r: httpx.Response(500)))
    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(client.publish("t", "m"))
