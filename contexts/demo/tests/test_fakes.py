"""Per-context fakes: every event is a real contract event, and together they cover what the contexts publish."""

import asyncio
import json

import pytest

from bus import EventBus
from contexts.demo import fakes
from contexts.demo.__main__ import main
from contexts.demo.story import load
from contracts import REGISTRY, parse_event

START = load().now


@pytest.mark.parametrize("context", fakes.CONTEXTS)
def test_fake_publishes_every_type_its_context_owns_and_nothing_else(context, tmp_path):
    published = asyncio.run(fakes.publish(context, tmp_path, START))
    assert published and {e.type for e in published} == fakes.owned(context)

    lines = [line for path in sorted(tmp_path.glob("*.jsonl")) for line in path.read_text().splitlines()]
    assert len(lines) == len(published)
    for line in lines:  # what reached the log parses back into its registered class
        event = parse_event(line)
        assert type(event) is REGISTRY[json.loads(line)["type"]]
    assert len({e.event_id for e in published}) == len(published)


def test_fakes_cover_every_context_that_publishes_events_to_the_phone():
    covered = set().union(*(fakes.owned(c) for c in fakes.CONTEXTS))
    others = {t.split(".")[0] for t in set(REGISTRY) - covered}
    assert others == {"capture", "experience", "learning", "runtime", "telemetry", "workspace"}  # not asked for in F4's fake command


def test_fake_is_the_same_events_every_time_so_a_replay_drops_the_repeat(tmp_path):
    first = asyncio.run(fakes.publish("habits", tmp_path, START))
    second = asyncio.run(fakes.publish("habits", tmp_path, START))
    assert [e.model_dump_json() for e in first] == [e.model_dump_json() for e in second]
    bus = EventBus(tmp_path)
    assert asyncio.run(bus.replay_into()) == 2 * len(first)  # read twice, handled once
    seen = []
    bus = EventBus(tmp_path)
    bus.subscribe("habits.", seen.append, replay=True)
    asyncio.run(bus.replay_into())
    assert len(seen) == len(first)


def test_fake_command(tmp_path, capsys):
    assert main(["fake", "impact", "--data-dir", str(tmp_path)]) == 0
    assert "impact.count.updated" in capsys.readouterr().out
    assert list((tmp_path / "events").glob("*.jsonl"))


def test_after_the_fake_scene_the_keys_still_have_a_place_and_a_frame(tmp_path):
    """python -m contexts.demo fake perception --gateway ... into a box that already knows where the keys are (a seeded
    night): the scene's sighting is later, so it is the last one, and it still says where on the desk and cites a frame."""
    import re
    from datetime import UTC, datetime, timedelta

    from fastapi.testclient import TestClient

    from contexts.gateway.app import INPUT_EVENTS, create_app
    from contexts.gateway.bootstrap import Settings

    start = datetime.now(UTC).replace(microsecond=0)
    before = start - timedelta(hours=9)
    seeded = {"type": "perception.scene.described", "ts": before.isoformat(), "camera_id": "cam-desk", "zone_id": "desk",
              "caption": "On the desk: keys (left of the laptop, behind the coffee mug), laptop.", "model_version": "seed",
              "objects": [{"label": "keys", "position": "left of the laptop, behind the coffee mug"}],
              "frame_ref": f"frames:cam-desk:{int(before.timestamp() * 1000)}"}
    events = asyncio.run(fakes.publish("perception", tmp_path / "scratch", start))
    pushed = sorted((e for e in events if e.type.startswith(INPUT_EVENTS)), key=lambda e: e.ts)
    settings = Settings(data_dir=tmp_path / "data", connectors=False, ntfy_url=None, llm_url=None, memory_fixture=None)
    with TestClient(create_app(settings)) as c:
        assert c.post("/api/events", json=[seeded]).status_code == 200
        was = c.get("/memory/objects/keys/last-seen").json()
        assert (was["position"], was["frame_ref"]) == (seeded["objects"][0]["position"], seeded["frame_ref"])
        assert c.post("/api/events", json=[e.model_dump(mode="json", exclude={"event_id"}) for e in pushed]).status_code == 200
        now = c.get("/memory/objects/keys/last-seen").json()
        assert datetime.fromisoformat(now["ts"]) > start  # the scene is the later sighting: it is the last one
        assert (now["camera_id"], now["zone_id"]) == ("cam-desk", "desk") and now["position"]  # a place, in words
        ref = re.fullmatch(r"frames:cam-desk:(\d{13})", now["frame_ref"])  # and a frame of that moment, on that camera
        assert ref and abs(int(ref[1]) / 1000 - datetime.fromisoformat(now["ts"]).timestamp()) < 1
        answer = c.post("/api/ask", json={"text": "Where did I leave my keys?"}).json()
        assert now["position"] in answer["answer"] and "desk" in answer["answer"]
        assert any(x.get("frame_ref") == now["frame_ref"] for x in answer["citations"])
