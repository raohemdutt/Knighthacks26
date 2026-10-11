"""The replay: event-time order, the live picture, seek, speed, and rebuilding the picture from the log after a restart."""

import asyncio
from pathlib import Path

import pytest

from bus import EventBus
from contexts.spot.application.replay import MAX_WAIT_S, Player, SpotLive, build_stream, events_for
from contexts.spot.infrastructure.kit import load_kit
from contexts.spot.tests.helpers import small_kit


@pytest.fixture(scope="module")
def kit():
    return load_kit(Path("fixtures/spot/kit"))


def wire(kit, log_dir, persist=True):
    bus = EventBus(log_dir, persist=persist)
    stream = build_stream(kit)
    live = SpotLive(kit, stream)
    bus.subscribe("spot.", live.on_event, replay=True)
    seen = []
    bus.subscribe("spot.", seen.append)

    async def no_sleep(_):
        await asyncio.sleep(0)

    return bus, live, Player(kit, stream, bus, live, sleep=no_sleep), seen


def test_stream_is_in_event_time_order_and_completed_from_the_tables(kit):
    stream = build_stream(kit)
    assert all(a.timestamp <= b.timestamp for a, b in zip(stream, stream[1:]))
    assert [e.event_id for e in stream[:500]] == [e.event_id for e in kit.replay]  # the kit's own lines, untouched and first
    rebuilt = [e for e in stream if e.source == "tables"]
    assert len(stream) == 594 and len(rebuilt) == 94 and {e.mission_id for e in rebuilt} == {"M023", "M024", "M025", "M026", "M027"}
    # every mission starts once and ends once, every action starts once and ends once
    assert sum(e.event_type == "MISSION_STARTED" for e in stream) == sum(e.event_type == "MISSION_COMPLETED" for e in stream) == 27
    ends = [e.details["action_id"] for e in stream if e.event_type in ("ACTION_COMPLETED", "ACTION_FAILED")]
    assert sorted(ends) == sorted(a.action_id for a in kit.actions)
    for mid in ("M023", "M027"):  # inside a mission: start, then each action's start before its end, then the mission's end
        kinds = [e.event_type for e in stream if e.mission_id == mid]
        assert kinds[0] == "MISSION_STARTED" and kinds[-1] == "MISSION_COMPLETED" and kinds[1:-1:2] == ["ACTION_STARTED"] * 10


def test_a_failed_action_and_its_anomaly_raise_alerts_at_that_moment(kit):
    stream = build_stream(kit)
    i = next(i for i, e in enumerate(stream) if e.event_type == "ACTION_FAILED" and e.details["action_id"] == "A0001")
    kinds = [e.type for e in events_for(kit, stream[i], i)]
    assert kinds == ["spot.action.failed", "spot.alert.raised", "spot.anomaly.raised", "spot.alert.raised"]  # AN024 sits on A0001
    out = events_for(kit, stream[i], i)
    assert out[0].failure_type == "OBSTACLE_DETECTION" and out[0].reading["quality"] == "QUESTIONABLE"
    assert out[2].anomaly_id == "AN024" and out[2].confirmed is False and "unverified" in out[3].message
    assert all(e.data_ts == "2026-09-01T18:03:06" and e.seq == i for e in out)


def test_playing_builds_the_picture_and_flags_a_completed_mission_with_failures(kit, tmp_path):
    bus, live, player, seen = wire(kit, tmp_path)

    async def run():
        for _ in range(22):  # all of M001
            assert await player.step()

    asyncio.run(run())
    v = live.view()
    assert v["index"] == 22 and v["mission"]["mission_id"] == "M001" and v["mission"]["status"] == "COMPLETED"
    assert v["mission"]["failed"] == 2 and [a["status"] for a in v["mission"]["actions"]].count("FAILED") == 2
    assert v["counts"]["missions_hiding_failures"] == 1 and v["counts"]["actions_ok"] == 8 and v["counts"]["actions_failed"] == 2
    assert v["robot"]["location_id"] == "WP05" and v["robot"]["floor"] == 1 and v["robot"]["trail"][0] == "WP05"
    assert v["by_location"]["WP05"] == {"ok": 1, "failed": 1}
    assert v["alerts"][0]["message"] == "M001 ended COMPLETED with 2 failed action(s) inside."
    done = [e for e in seen if e.type == "spot.mission.completed"][0]
    assert done.failed_actions == 2 and done.status == "COMPLETED"
    assert v["rebuilt_from"] == 500 and not v["at_end"]


def test_seek_gives_the_same_picture_as_playing_there_and_can_go_back(kit, tmp_path):
    bus, live, player, _ = wire(kit, tmp_path / "a")
    other_bus, other_live, other_player, _ = wire(kit, tmp_path / "b")

    async def run():
        for _ in range(137):
            await player.step()
        await other_player.seek(137)
        assert other_live.view() == live.view()
        await other_player.seek(30)
        assert other_live.index == 30 and other_live.view()["mission"]["mission_id"] == "M002"
        await other_player.seek(0)
        assert other_live.view()["mission"] is None and other_live.view()["counts"]["actions_ok"] == 0
        await other_player.seek(10_000)
        assert other_live.view()["at_end"] and other_live.view()["counts"]["missions_started"] == 27

    asyncio.run(run())
    end = other_live.view()["counts"]
    assert end["actions_ok"] == 221 and end["actions_failed"] == 49 and end["anomalies"] == 40 and end["missions_hiding_failures"] == 21


def test_a_restart_rebuilds_the_picture_from_the_log(kit, tmp_path):
    bus, live, player, _ = wire(kit, tmp_path)

    async def before():
        await player.seek(40)
        for _ in range(25):
            await player.step()

    asyncio.run(before())
    picture = live.view()
    assert picture["index"] == 65

    bus2, live2, player2, seen2 = wire(kit, tmp_path)  # a new process on the same log
    assert live2.index == 0
    replayed = asyncio.run(bus2.replay_into())
    assert replayed > 25 and live2.view() == picture
    assert seen2 == []  # live-only subscribers saw nothing: no alert fires twice
    assert not player2.playing  # it comes back paused where it was


def test_play_pause_speed_and_running_to_the_end():
    kit = small_kit()  # no replay file at all: the whole stream is rebuilt from the tables
    bus, live, player, seen = wire(kit, "unused", persist=False)
    assert len(player.stream) == 3 * 22 and all(e.source == "tables" for e in player.stream)

    async def run():
        await player.start()
        assert player.playing
        await player.pause()
        assert not player.playing
        at = live.index
        await asyncio.sleep(0.01)
        assert live.index == at  # paused means paused
        await player.start()
        for _ in range(2000):
            if not player.playing:
                break
            await asyncio.sleep(0)
        assert live.view()["at_end"] and not player.playing
        await player.start()  # at the end: plays again from the top
        await player.pause()
        assert live.index < len(player.stream)

    asyncio.run(run())
    assert live.counts["missions_failed"] == 1 or live.index < len(player.stream)
    assert player.set_speed(240) == 240 and player.set_speed(0.001) == 1.0
    assert sum(e.type == "spot.alert.raised" and e.level == "critical" for e in seen) >= 1  # M002 ends FAILED


def test_waits_follow_event_time_scaled_by_speed_with_long_gaps_cut(kit):
    stream = build_stream(kit)
    player = Player(kit, stream, EventBus("unused", persist=False), SpotLive(kit, stream), speed=60)
    second_start = next(i for i, e in enumerate(stream) if e.event_type == "ACTION_STARTED" and e.details["action_id"] == "A0002")
    assert player.wait_before(second_start) == pytest.approx(min(186 / 60, MAX_WAIT_S))  # 3 min 6 s of data
    player.set_speed(1200)
    assert player.wait_before(second_start) == pytest.approx(186 / 1200)
    next_mission = next(i for i, e in enumerate(stream) if e.mission_id == "M002")
    assert player.wait_before(next_mission) == MAX_WAIT_S  # seven and a half idle hours are not waited out
    assert player.wait_before(0) == 0


def test_without_a_kit_nothing_plays():
    bus = EventBus("unused", persist=False)
    live = SpotLive(None)
    player = Player(None, [], bus, live)

    async def run():
        await player.start()
        assert not await player.step()
        await player.seek(5)

    asyncio.run(run())
    assert live.view()["index"] == 0 and live.view()["total"] == 0 and live.marks() == []
