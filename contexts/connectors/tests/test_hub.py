import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from contexts.connectors.application.hub import ConnectorHub
from contexts.connectors.domain.spec import ConnectorSpec, load_specs
from contexts.connectors.fakes.fake import FakeConnector
from contexts.connectors.infrastructure.mcp_client import McpConnector
from contexts.workspace.domain.approvals import Approvals
from contracts.connectors import ItemSynced, SyncFailed
from contracts.experience import ApprovalDecided

T0 = datetime(2026, 10, 10, 12, 0, tzinfo=UTC)
ASSIGNMENT = {"external_id": "a1", "title": "HW 3", "when": "2026-10-12T23:59:00-04:00", "url": "https://canvas/a1"}
SPEC = ConnectorSpec(name="canvas", module="x", every_s=60, reads={"list_items": "assignment"}, writes=["post"])


def hub_with(fake, offline=False):
    return ConnectorHub([SPEC], factory=lambda spec: fake, offline=lambda: offline)


def test_sync_publishes_new_and_changed_items_only():
    fake = FakeConnector({"list_items": [ASSIGNMENT]})
    hub = hub_with(fake)
    [item] = asyncio.run(hub.sync("canvas", T0))
    assert isinstance(item, ItemSynced)
    assert (item.connector, item.kind, item.external_id, item.title) == ("canvas", "assignment", "a1", "HW 3")
    assert asyncio.run(hub.sync("canvas", T0)) == []
    fake.reads["list_items"] = [{**ASSIGNMENT, "title": "HW 3 (due date moved)"}]
    [changed] = asyncio.run(hub.sync("canvas", T0))
    assert changed.title == "HW 3 (due date moved)"


def test_schedule_failure_and_offline():
    fake = FakeConnector({"list_items": []})
    hub = hub_with(fake)
    assert hub.due(T0) == ["canvas"]
    asyncio.run(hub.sync_due(T0))
    assert hub.due(T0 + timedelta(seconds=30)) == []
    assert hub.due(T0 + timedelta(seconds=60)) == ["canvas"]

    fake.fail = "401 Unauthorized"
    [failed] = asyncio.run(hub.sync("canvas", T0))
    assert isinstance(failed, SyncFailed) and "401" in failed.error

    [paused] = asyncio.run(hub_with(FakeConnector(), offline=True).sync("canvas", T0))
    assert paused.offline


def test_writes_only_run_after_approval():
    fake = FakeConnector()
    hub = hub_with(fake)
    with pytest.raises(ValueError):
        hub.propose_write("canvas.list_items", {}, "not a write", T0)
    proposal = hub.propose_write("canvas.post", {"text": "study at 2?"}, "Plan the session", T0)
    assert proposal.kind == "connector_write" and proposal.target == "canvas.post"

    approvals = Approvals({"connector_write": hub.execute_write})
    approvals.on_proposed(proposal)
    assert fake.calls == []
    [done] = asyncio.run(approvals.on_decided(ApprovalDecided(ts=T0, action_id=proposal.action_id, decision="approve")))
    assert done.status == "done"
    assert fake.calls == [("post", {"text": "study at 2?"})]


def test_config_loads():
    specs = {s.name: s for s in load_specs("config/connectors.yaml")}
    assert specs["site"].reads == {"check_pages": "page_change"}
    assert not specs["calendar"].enabled  # stays off until the feed URLs are in .env


def test_real_mcp_round_trip():
    spec = ConnectorSpec(
        name="fake",
        module="contexts.connectors.servers.fake_server",
        reads={"list_items": "post"},
        writes=["post"],
        options={"items": [{"external_id": "p1", "title": "hello"}]},
    )

    async def run():
        hub = ConnectorHub([spec], factory=McpConnector)
        try:
            [item] = await hub.sync("fake", T0)
            proposal = hub.propose_write("fake.post", {"text": "hi"}, "test", T0)
            approvals = Approvals({"connector_write": hub.execute_write})
            approvals.on_proposed(proposal)
            [done] = await approvals.on_decided(ApprovalDecided(ts=T0, action_id=proposal.action_id, decision="approve"))
            return item, done
        finally:
            await hub.close()

    item, done = asyncio.run(run())
    assert item.title == "hello"
    assert (done.status, done.result) == ("done", "posted: hi")


def test_on_demand_tools_are_declared_reads_only():
    spec = ConnectorSpec(name="browser", module="x", tools=["search"], writes=["submit"])
    fake = FakeConnector({"search": [{"title": "t", "url": "u"}]})
    hub = ConnectorHub([spec], factory=lambda s: fake)
    assert asyncio.run(hub.call_tool("browser.search", {"query": "q"})) == [{"title": "t", "url": "u"}]
    with pytest.raises(ValueError):
        asyncio.run(hub.call_tool("browser.submit", {}))  # writes must be proposed, never called directly
    assert fake.calls == [("search", {"query": "q"})]
