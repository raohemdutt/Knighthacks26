"""The demo night's second League member: a friend's shared daily summaries, as local events.

    events = friend_summaries(now)   # play.league.summary.received, one per habit per day

Demo fixture only (fixtures/demo/league.yaml): what Maya's box would have shared to the Circle's
habit_summaries and impact_counters, written relative to the story's end so `--now` moves it too.
Deterministic ids, so seeding twice adds nothing. Nothing here touches the owner's own habits.
"""

from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml
from pydantic import BaseModel, ConfigDict

from contracts.events import Event
from contracts.play import LeagueSummaryReceived

ROOT = Path(__file__).resolve().parents[3]
FIXTURE = ROOT / "fixtures" / "demo" / "league.yaml"
LOCAL = ZoneInfo("America/New_York")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FriendDay(_Strict):
    days_ago: int  # 1 = the story's last full day
    habits: dict[str, float]  # habit id -> the day's value
    met: dict[str, bool] = {}  # when it differs from the habit's own direction, say so


class FriendCounts(_Strict):
    items_seen: int
    recyclable_seen: int
    recycled: int
    contamination: int


class Friend(_Strict):
    member_id: str
    display_name: str
    days: list[FriendDay]
    counts: FriendCounts | None = None


class Fixture(_Strict):
    circle_id: str
    friends: list[Friend]


# The owner's habits (config/habits.yaml): what "met" means for each, so the fixture needs only values.
DIRECTION = {"sleep": (">=", 6), "meals": (">=", 2), "energy_drinks": ("<=", 2), "focus": (">=", 2),
             "leave_ready": ("<=", 0), "sort_right": (">=", 80)}


def load(path: str | Path = FIXTURE) -> Fixture:
    return Fixture.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def friend_summaries(now: datetime, circle_id: str | None = None, fixture: Fixture | None = None) -> list[Event]:
    """Every friend's rows as events, oldest first. `circle_id` overrides the fixture's (the local league's id)."""
    fx = fixture or load()
    cid = circle_id or fx.circle_id
    today = now.astimezone(LOCAL).date()
    out: list[Event] = []
    for f in fx.friends:
        for d in f.days:
            day = today - timedelta(days=d.days_ago)
            ts = datetime.combine(day + timedelta(days=1), datetime.min.time(), LOCAL) + timedelta(minutes=5)
            for habit, value in sorted(d.habits.items()):
                op, target = DIRECTION.get(habit, (">=", 0))
                met = d.met.get(habit, value >= target if op == ">=" else value <= target)
                out.append(LeagueSummaryReceived(
                    event_id=f"demo-league-{f.member_id}-{habit}-{day.isoformat()}", ts=ts, circle_id=cid,
                    member_id=f.member_id, kind="habit_summary", display_name=f.display_name,
                    payload={"habit_id": habit, "day": day.isoformat(), "met": met, "value": value, "streak": 0}))
        if f.counts:
            c = f.counts
            ts = datetime.combine(today, datetime.min.time(), LOCAL) + timedelta(hours=1)
            out.append(LeagueSummaryReceived(
                event_id=f"demo-league-{f.member_id}-counts-{today.isoformat()}", ts=ts, circle_id=cid,
                member_id=f.member_id, kind="impact_count", display_name=f.display_name,
                payload={**c.model_dump(), "diversion_rate": round(c.recycled / c.recyclable_seen, 3) if c.recyclable_seen else 0}))
    return sorted(out, key=lambda e: (e.ts, e.event_id))
