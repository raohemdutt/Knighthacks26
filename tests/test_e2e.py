"""End-to-end smoke test of the whole app (D1a).

Drives the real gateway (bootstrap, bus, every context) with the scripted camera
feed from contexts.perception.fakes and checks the demo beats that need no
hardware: zones, the package alert, Waste Watch, Ask, and a restart that replays
the log to the same state without re-sending anything.
"""

from datetime import UTC, datetime, timedelta

from fastapi.testclient import TestClient

from contexts.gateway.app import create_app
from contexts.gateway.bootstrap import Settings
from contexts.perception.fakes.script import script

START = datetime(2026, 10, 10, 13, 0, tzinfo=UTC)


def app(tmp_path) -> TestClient:
    return TestClient(create_app(Settings(data_dir=tmp_path, connectors=False, ntfy_url=None, llm_url=None, token=None)))


def test_scripted_scene_through_the_whole_app(tmp_path):
    seen: list = []
    with app(tmp_path) as c:
        atlas = c.app.state.atlas
        atlas.bus.subscribe("", seen.append)

        async def play():
            await atlas.bus.publish_all(script(START))
            await atlas.bus.publish_all(atlas.zone_engine.flush(START + timedelta(minutes=5)))

        c.portal.call(play)

        types = [e.type for e in seen]
        zone_events = [e for e in seen if e.type == "perception.zone.event"]
        assert {(e.label, e.zone_id) for e in zone_events} >= {("person", "door"), ("person", "desk"), ("package", "door"), ("can", "recycling")}

        [alert] = [e for e in seen if e.type == "agent.alert.raised"]
        assert alert.rule_id == "package-at-door"
        assert any(e.type == "impact.item.disposed" and e.correct for e in seen)
        assert c.get("/api/impact").json()["total"]["recycled"] == 1

        r = c.post("/api/ask", json={"text": "What is due this week?"}).json()
        assert r["citations"] and all(cit["url"] for cit in r["citations"])
        assert c.post("/api/ask", json={"text": "Where is my wallet?"}).json()["answer"] == "I didn't see that."
        assert types.count("agent.alert.raised") == 1

    # Restart: the log replays into the state that rebuilds, and nothing is raised or sent twice.
    again: list = []
    with app(tmp_path) as c:
        atlas = c.app.state.atlas
        assert c.get("/api/impact").json()["total"]["recycled"] == 1
        atlas.bus.subscribe("", again.append)
        c.portal.call(lambda: atlas.bus.publish_all(script(START)))  # the same observations again: deduped by id
        assert not [e for e in again if e.type == "agent.alert.raised"]
