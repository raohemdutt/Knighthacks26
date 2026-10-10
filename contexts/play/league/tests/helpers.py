"""Fakes for the League tests: no network, no Neon, no connectors."""

from datetime import date, datetime, timedelta
from uuid import uuid4
from zoneinfo import ZoneInfo

from contexts.habits.domain.config import Schedule
from contexts.play.league.application.league import League
from contexts.play.league.config import load_config
from contracts.agent import ActionProposed
from contracts.habits import DayClosed
from contracts.play import LeagueSummaryReceived

TZ = ZoneInfo("America/New_York")
SCHED = Schedule(tz=TZ)
MONDAY = date(2026, 10, 5)  # the week under test: Mon Oct 5 to Sun Oct 11
NAMES = {"hemdutt": "Hemdutt", "maya": "Maya", "dom": "Dominick"}


def at(day: date, hour: int = 12, minute: int = 0) -> datetime:
    return datetime(day.year, day.month, day.day, hour, minute, tzinfo=TZ)


def league(circle_id="team", me="hemdutt", names=NAMES, hub=None, circle=None, cfg_path=None, clock=None) -> League:
    return League(load_config(cfg_path), circle_id, me, names, SCHED, hub=hub, circle=circle, clock=clock)


def day_closed(habit: str, day: date, met: bool, value: float, member: str = "hemdutt") -> DayClosed:
    """This box's own summary (the event has no member; the league files it under `me`)."""
    return DayClosed(event_id=f"habits-day-{habit}-{day}", ts=at(day + timedelta(days=1), 0, 2), habit_id=habit, day=day,
                     met=met, value=value, streak=0)


def shared(member: str, habit: str, day: date, met: bool, value: float, circle_id="team", name=None) -> LeagueSummaryReceived:
    return LeagueSummaryReceived(event_id=f"sum-{circle_id}-{member}-{habit}-{day}", ts=at(day + timedelta(days=1), 0, 10),
                                 circle_id=circle_id, member_id=member, kind="habit_summary", display_name=name,
                                 payload={"habit_id": habit, "day": day.isoformat(), "met": met, "value": value, "streak": 0})


def week_of_sleep(lg: League, member: str, met_nights: int, circle_id="team") -> list:
    """Seven sleep summaries, the first `met_nights` met. Mine go in as DayClosed, a friend's as shared rows."""
    out = []
    for i in range(7):
        day, met = MONDAY + timedelta(days=i), i < met_nights
        value = 7.0 if met else 4.5
        e = day_closed("sleep", day, met, value) if member == lg.me else shared(member, "sleep", day, met, value, circle_id)
        out.append(e)
    return out


class FakeHub:
    """Declares groupme.post as a write; never calls anything."""

    def __init__(self, writes=("groupme.post",)):
        self.writes = set(writes)
        self.calls: list = []

    def is_write(self, tool: str) -> bool:
        return tool in self.writes

    def propose_write(self, tool, arguments, reason, now):
        return ActionProposed(ts=now, action_id=uuid4().hex, kind="connector_write", target=tool, reason=reason, params=arguments)


class FakeCircle:
    def __init__(self, circle_id="team", member_id="hemdutt"):
        self.circle_id, self.member_id = circle_id, member_id
        self.posted: list = []

    def propose(self, proposal, reason):
        return ActionProposed(ts=proposal.start, action_id=uuid4().hex, kind="circle_share", target=self.circle_id, reason=reason,
                              params={"kind": "event_proposal", "thread_id": uuid4().hex, "proposal": proposal.model_dump(mode="json")})

    def share_executor(self, req) -> str:
        self.posted.append(req)
        return "proposed to the circle"
