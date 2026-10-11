"""Per-context fakes (F4): a small realistic sample of every event type one context publishes.

    python -m contexts.demo fake habits        # appended to data/events, replayed by the gateway when it starts

For building a screen against one context without the rest of the app. The steps are data
(fixtures/demo/fakes.yaml): each one is that type's sample payload from
tests/test_contract_samples.py with a few fields changed. Ids and times follow from the start
time, so publishing the same fake twice gives the same events and a replay drops the repeats.
"""

import re
from datetime import datetime, timedelta
from pathlib import Path

import yaml

from bus import EventBus
from contexts.demo.story import ROOT, ZONES
from contracts import REGISTRY, parse_event
from contracts.events import Event

STEPS = ROOT / "fixtures" / "demo" / "fakes.yaml"
# context -> prefix of the event types it publishes
PREFIX = {"perception": "perception.", "connectors": "connector.", "memory": "memory.", "habits": "habits.",
          "impact": "impact.", "agent": "agent.", "circle": "circle.", "vault": "vault.", "play": "play.", "day": "day."}
CONTEXTS = tuple(PREFIX)
_OFFSET = re.compile(r"^[+-]\d+s$")


def owned(context: str) -> set[str]:
    """The registered event types `context` publishes."""
    return {t for t in REGISTRY if t.startswith(PREFIX[context])}


def _samples() -> dict[str, dict]:
    try:
        from tests.test_contract_samples import SAMPLES  # one payload per event type, kept valid by its own test
    except ImportError as err:  # only without the repo's tests or the dev dependencies
        raise RuntimeError("the fakes are built from tests/test_contract_samples.py: run from the repo root "
                           'with the dev dependencies installed (pip install -e ".[dev]")') from err
    return SAMPLES


def _timed(value, start: datetime):
    """"+90s" anywhere in a payload is a time 90 seconds after the start."""
    if isinstance(value, str) and _OFFSET.match(value):
        return (start + timedelta(seconds=int(value[:-1]))).isoformat()
    if isinstance(value, dict):
        return {k: _timed(v, start) for k, v in value.items()}
    if isinstance(value, list):
        return [_timed(v, start) for v in value]
    return value


def events(context: str, start: datetime, path: str | Path = STEPS) -> list[Event]:
    """The context's steps as typed events, in the order they are published."""
    steps = (yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}).get(context) or []
    samples = _samples()
    out = []
    for i, step in enumerate(steps):
        step = dict(step)
        type_name, ts = step.pop("type"), start + timedelta(seconds=step.pop("at", 0))
        payload = samples[type_name] | _timed(step, start)
        out.append(parse_event({"event_id": f"fake-{context}-{i}-{int(start.timestamp())}", "type": type_name,
                                "ts": ts.isoformat(), **payload}))
    return out


async def publish(context: str, log_dir: str | Path, start: datetime) -> list[Event]:
    """Append the context's fake to the log in `log_dir`. Returns what was published, derived zone events included."""
    bus = EventBus(log_dir)
    published: list[Event] = []
    bus.subscribe("", published.append)
    if context == "perception":  # the scripted camera scene, with the zone events the real engine makes of it
        from contexts.perception.domain.zones import ZoneEngine
        from contexts.perception.fakes.script import ROOM_LENGTH_S, desk_described, play, room_scene

        engine = ZoneEngine.load(ZONES)
        bus.subscribe("perception.observation", engine.on_observation)
        await play(bus, start, scene=room_scene)  # the half minute of script(), then a Sort It round for the room twin
        await bus.publish(desk_described(start))  # where the keys lie at the end: the last sighting keeps a place and a frame
        await bus.publish_all(engine.flush(start + timedelta(seconds=ROOM_LENGTH_S) + engine.exit_after))
    await bus.publish_all(events(context, start))
    return published
