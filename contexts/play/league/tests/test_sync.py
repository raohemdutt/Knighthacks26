"""LeagueSync on a fake remote: friends' rows become events, mine are skipped, declarations are pushed once."""

import asyncio
from datetime import timedelta

from bus import EventBus
from contexts.play.league.application.sync import LeagueSync
from contexts.play.league.tests.helpers import MONDAY, at, league, week_of_sleep
from contracts.play import Stake


class FakeRemote:
    def __init__(self):
        self.rows = []
        self.counters = []
        self.pushed = []

    def habit_summaries(self, circle_id, since):
        return [r for r in self.rows if r["day"] >= since]

    def impact_counters(self, circle_id):
        return self.counters

    def push_stake(self, e):
        self.pushed.append(("stake", e.week_start, e.stake.kind))

    def push_week(self, e):
        self.pushed.append(("week", e.week_start))

    def push_challenge(self, started, closed):
        self.pushed.append(("challenge", started.challenge_id, closed is not None))


def test_friends_rows_become_events_once_and_mine_never_do(tmp_path):
    bus = EventBus(tmp_path / "events")
    lg = league()
    lg.subscribe(bus)
    remote = FakeRemote()
    remote.rows = [{"member_id": m, "habit_id": "sleep", "day": MONDAY, "met": m == "hemdutt", "value": 7 if m == "hemdutt" else 4, "streak": 1}
                   for m in ("hemdutt", "maya")]
    remote.counters = [{"member_id": "maya", "items_seen": 3, "recyclable_seen": 2, "recycled": 1, "contamination": 0,
                        "diversion_rate": 0.5, "updated_at": at(MONDAY, 10)}]
    sync = LeagueSync(remote, lg)
    now = at(MONDAY + timedelta(days=1), 9)
    out = sync.sync(now)
    assert [(e.member_id, e.kind) for e in out] == [("maya", "habit_summary"), ("maya", "impact_count")]
    assert out[0].payload == {"habit_id": "sleep", "day": "2026-10-05", "met": False, "value": 4.0, "streak": 1}
    assert out[1].ts == at(MONDAY, 10) and out[1].payload["diversion_rate"] == 0.5
    asyncio.run(bus.publish_all(out))
    assert sync.sync(now) == []  # the league knows them now
    remote.rows[1]["met"], remote.rows[1]["value"] = True, 6.5  # the friend's box corrected the day
    [again] = sync.sync(now)
    assert again.payload["value"] == 6.5 and again.event_id != out[0].event_id


def test_declarations_are_pushed_once_each(tmp_path):
    bus = EventBus(tmp_path / "events")
    lg = league()
    lg.subscribe(bus)
    for e in week_of_sleep(lg, "hemdutt", 6) + week_of_sleep(lg, "maya", 2):
        asyncio.run(bus.publish(e))
    lg.stakes[MONDAY] = Stake(kind="picks_the_event")
    asyncio.run(bus.publish_all(lg.flush(at(MONDAY + timedelta(days=7), 4))))
    remote = FakeRemote()
    sync = LeagueSync(remote, lg)
    sync.sync(at(MONDAY + timedelta(days=7), 5))
    assert remote.pushed == [("stake", MONDAY, "picks_the_event"), ("week", MONDAY)]
    sync.sync(at(MONDAY + timedelta(days=7), 6))
    assert len(remote.pushed) == 2
