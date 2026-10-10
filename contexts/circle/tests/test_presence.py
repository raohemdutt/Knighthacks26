import asyncio
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from contexts.agent.application.memory_port import FakeMemory
from contexts.agent.application.planner import Planner
from contexts.agent.application.qa import QA, SleepSchedule
from contexts.circle.application.presence import PresenceBoard
from contracts.circle import PlaceShared
from contracts.memory import Item
from contracts.perception import ZoneEvent
from runtime import FakeRuntime

NY = ZoneInfo("America/New_York")
NOW = datetime(2026, 10, 10, 19, tzinfo=NY)
PLACES = ["Our table", "Library", "Dorm"]


def board(opted_in=True):
    return PresenceBoard("team", "dom", PLACES, opted_in)


def test_only_named_places_and_only_when_opted_in():
    assert board().share("Library", "phone", NOW).place == "Library"
    with pytest.raises(ValueError):
        board().share("28.6024,-81.2001", "phone", NOW)  # coordinates never pass
    with pytest.raises(PermissionError):
        board(opted_in=False).share("Library", "phone", NOW)


def test_desk_presence_shares_our_table_once_an_hour():
    b = board()
    sit = ZoneEvent(ts=NOW, camera_id="cam-desk", track_id=1, label="person", zone_id="desk", kind="enter")
    [shared] = b.on_zone_event(sit)
    assert (shared.place, shared.source) == ("Our table", "desk")
    b.on_shared(shared)
    assert b.on_zone_event(sit.model_copy(update={"ts": NOW + timedelta(minutes=10)})) == []
    assert board(opted_in=False).on_zone_event(sit) == []


def test_friends_expire_and_where_should_i_go_uses_them():
    b = board()
    b.on_shared(PlaceShared(ts=NOW, circle_id="team", member_id="hemdutt", place="Library", source="phone", expires_at=NOW + timedelta(hours=2)))
    b.on_shared(PlaceShared(ts=NOW, circle_id="team", member_id="xander", place="Dorm", source="phone", expires_at=NOW - timedelta(minutes=1)))
    assert [p.member_id for p in b.friends(NOW)] == ["hemdutt"]

    event = Item(connector="calendar", kind="calendar_event", external_id="e1", title="Knight Hacks midnight snack", when=NOW + timedelta(hours=5))
    rt = FakeRuntime()
    qa = QA(rt, FakeMemory(items=[event]), clock=lambda: NOW, sleep=SleepSchedule(tz=NY))
    planner = Planner(qa)
    planner.presence = b
    qa.planner = planner
    asyncio.run(qa.ask("Where should I go tonight?"))
    prompt = rt.calls[-1][1][-1].content
    assert "hemdutt is at Library (shared until 9:00 PM)" in prompt and "midnight snack" in prompt and "xander" not in prompt
