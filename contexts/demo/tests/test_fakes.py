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
    assert others == {"capture", "experience", "learning", "runtime", "workspace"}  # not asked for in F4's fake command


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
