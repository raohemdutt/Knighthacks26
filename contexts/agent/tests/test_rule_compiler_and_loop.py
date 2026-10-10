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
    [alert] = asyncio.run(loop.tick(now))
    assert alert.message == "Activity on cam-desk at 3:12 AM, during sleep hours"
    assert alert.evidence[0].frame_ref == "frames:cam-desk:n1"
    assert not loop.due(now + timedelta(minutes=5)) and loop.due(now + timedelta(minutes=10))
    assert asyncio.run(loop.tick(now + timedelta(minutes=10))) == []


def test_daytime_activity_is_normal():
    noon = datetime(2026, 10, 11, 12, 0, tzinfo=NY)
    ep = Episode(id="d1", camera_id="cam-desk", start=noon - timedelta(minutes=3), labels=["person"])
    alerts = [e for e in asyncio.run(AgentLoop(FakeMemory([ep]), SleepSchedule(tz=NY)).tick(noon)) if e.type == "agent.alert.raised"]
    assert alerts == []


def test_hourly_note_once_per_hour_from_the_model():
    from contracts.agent import NoteWritten

    at = datetime(2026, 10, 11, 15, 5, tzinfo=NY)
    eps = [Episode(id=f"h{i}", camera_id="cam-desk", start=datetime(2026, 10, 11, 14, 10 * i, tzinfo=NY), labels=["person"],
                   captions=[c]) for i, c in enumerate(["Two people code at the table.", "Pizza arrives.", "Two people code at the table."])]
    seen = []
    loop = AgentLoop(FakeMemory(eps), SleepSchedule(tz=NY), summarize=lambda caps: seen.append(caps) or "Coding, then pizza.")
    notes = [e for e in asyncio.run(loop.tick(at)) if isinstance(e, NoteWritten)]
    assert [(n.text, n.hour.hour, sorted(n.episode_ids)) for n in notes] == [("Coding, then pizza.", 14, ["h0", "h1", "h2"])]
    assert seen == [["Two people code at the table.", "Pizza arrives."]]  # deduped captions
    assert [e for e in asyncio.run(loop.tick(at + timedelta(minutes=20))) if isinstance(e, NoteWritten)] == []


def test_package_at_an_empty_home_proposes_a_message():
    from contracts.agent import AlertRaised

    t = datetime(2026, 10, 11, 15, 0, tzinfo=NY)
    alert = AlertRaised(ts=t, alert_id="a", rule_id="package-at-door", message="Package left at the door")
    empty = AgentLoop(FakeMemory([]), SleepSchedule(tz=NY), notify="dom")
    [proposal] = empty.on_alert(alert)
    assert (proposal.kind, proposal.target) == ("send_message", "dom") and "Package" in proposal.params["text"]
    home = AgentLoop(FakeMemory([Episode(id="p", camera_id="cam-desk", start=t - timedelta(minutes=5), labels=["person"])]),
                     SleepSchedule(tz=NY), notify="dom")
    assert home.on_alert(alert) == []
    assert AgentLoop(FakeMemory([]), SleepSchedule(tz=NY)).on_alert(alert) == []  # off unless configured
