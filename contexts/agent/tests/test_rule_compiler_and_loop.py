import asyncio
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from contexts.agent.application.loop import AgentLoop
from contexts.agent.application.memory_port import FakeMemory
from contexts.agent.application.qa import SleepSchedule
from contexts.agent.application.rule_compiler import RuleCompileError, compile_rule
from contexts.agent.domain.rules import RuleEvaluator
from contracts.agent import RuleAdded
from contracts.memory import Episode
from contracts.perception import ZoneEvent
from contracts.types import ChatResult
from runtime import FakeRuntime

NY = ZoneInfo("America/New_York")
ZONES = ["door", "desk"]


def runtime(reply: str) -> FakeRuntime:
    return FakeRuntime(chat_replies=[ChatResult(content=reply, model_version="v1")])


def test_plain_english_becomes_a_rule_that_fires():
    reply = 'Sure: {"id": "bag-on-desk", "when": {"label": ["backpack"], "zone": "desk", "kind": "dwell", "seconds": 60}, "notify": "Bag left on the desk", "cooldown_s": 300}'
    rule = asyncio.run(compile_rule("tell me if a backpack is left on the desk for a minute", runtime(reply), ZONES))
    ev = RuleEvaluator([])
    ev.on_rule_added(RuleAdded(rule=rule.model_dump(), source_text="..."))
    fired = ev.on_zone_event(ZoneEvent(camera_id="cam-desk", track_id=1, label="backpack", zone_id="desk", kind="dwell", dwell_s=61))
    assert [a.message for a in fired] == ["Bag left on the desk"]


@pytest.mark.parametrize("reply", [
    "I can't do that",
    '{"id": "x", "when": {"label": ["person"], "zone": "kitchen", "kind": "enter"}, "notify": "hi"}',
    '{"id": "x", "when": {"label": ["person"], "zone": "door", "kind": "teleport"}, "notify": "hi"}',
])
def test_bad_drafts_are_rejected(reply):
    with pytest.raises(RuleCompileError):
        asyncio.run(compile_rule("whatever", runtime(reply), ZONES))


def test_loop_flags_activity_during_sleep_hours_once():
    now = datetime(2026, 10, 11, 3, 20, tzinfo=NY)
    night = Episode(id="n1", camera_id="cam-desk", start=now - timedelta(minutes=8), labels=["person"], keyframe_refs=["frames:cam-desk:n1"])
    empty = Episode(id="n2", camera_id="cam-desk", start=now - timedelta(minutes=5), labels=[])
    loop = AgentLoop(FakeMemory([night, empty]), SleepSchedule(tz=NY))
    [alert] = loop.tick(now)
    assert alert.message == "Activity on cam-desk at 3:12 AM, during sleep hours"
    assert alert.evidence[0].frame_ref == "frames:cam-desk:n1"
    assert not loop.due(now + timedelta(minutes=5)) and loop.due(now + timedelta(minutes=10))
    assert loop.tick(now + timedelta(minutes=10)) == []


def test_daytime_activity_is_normal():
    noon = datetime(2026, 10, 11, 12, 0, tzinfo=NY)
    ep = Episode(id="d1", camera_id="cam-desk", start=noon - timedelta(minutes=3), labels=["person"])
    assert AgentLoop(FakeMemory([ep]), SleepSchedule(tz=NY)).tick(noon) == []
