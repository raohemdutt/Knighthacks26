"""Seed the Circle's SHARED database (Neon) with a few demo members, so the Circle screen's leaderboards, the
shared calendar and the League are not empty. `python -m contexts.demo circle` is the one-box demo and is not this.

    python -m contexts.demo neon --circle kh-judges                  # dry run: prints every row it would write
    python -m contexts.demo neon --circle kh-judges --apply          # writes them
    python -m contexts.demo neon --circle kh-judges --remove         # dry run of the removal
    python -m contexts.demo neon --circle kh-judges --remove --apply # deletes exactly what --apply wrote

What it writes, all of it through the classes a box itself writes with (so a row is what a box would have put there):

- per member and day, the habit summaries a box shares when a habit day closes, and a Waste Watch count
  (CircleSharing shapes the share, NeonShares.push lands it: `shares`, `habit_summaries`, `impact_counters`);
- two Sort It rounds per member (`shares`, `game_scores`);
- two plans the members agreed, "Dinner" at 6:30 PM and "Board game night" at 8:30 PM on the evening of --end:
  a proposal and each invitee's "free" (NeonCircleStore.post: `agent_messages`), then the agreed event
  (NeonCircleStore.record_event: `events`, `rsvps`). Every plan names its invitees, so no real box is asked;
- the League's declarations for the last complete week: the stake, the week's standings and a dare with its
  result (NeonLeague: `league_stakes`, `league_weeks`, `league_challenges`). They are computed by the real League
  projection from the summaries above, not written by hand.

The one row no store class can write is the circle with a name: a box names a circle after its id. The seed
writes it itself, with a name that is not the id, and that name is how it knows a circle is its own.

No places are written (`presence`): a place expires within hours and its row is rewritten on every push, so it
could be neither kept for judging nor "written once".

Everything is a function of --circle, --members, --days and the local DATE of --end (default: today), never of the
time of day: a second --apply on the same day adds nothing, and --remove with the same four values finds every
row again from another machine. --apply also keeps a manifest (<data-dir>/neon-seed/<circle>.json) so --remove
on this machine needs only --circle.

Safety:
- Member ids are global in the database (one `members` row per id, one habit row per member, habit and day), so a
  seed member must be the seed's alone. The default ids start with `demo-`. An id that already belongs to another
  circle, or that already has a row this run would not write, is refused: it is a real box's, or an earlier seed's.
- A circle that holds anything from someone else is refused unless --into-existing; with it only the seed's own
  members' rows are added, and a League declaration for a week that already has one is left alone.
- --remove deletes rows by their exact keys AND their owner, then a seed member only if nothing references it
  any more, then the circle only if the seed named it and it is empty. It never updates anything.
- The connection string comes from the environment or `.env` and is never printed.
"""

import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path

from contexts.circle.application.protocol import AgentMessage
from contexts.circle.application.sharing import CircleOffline, CircleSharing, OutboxRow
from contexts.circle.infrastructure.neon import CircleBoard, NeonCircleStore, NeonDb, NeonShares, NeonTransport
from contexts.demo.story import ROOT
from contexts.habits.domain.config import Schedule
from contexts.play.league import League, load_config as load_league
from contexts.play.league.demo import DIRECTION
from contexts.play.league.domain import scoring
from contexts.play.league.infrastructure.neon import NeonLeague
from contracts.circle import Proposal, Reply
from contracts.habits import DayClosed
from contracts.impact import CountUpdated
from contracts.play import (
    LeagueChallengeClosed,
    LeagueChallengeStarted,
    LeagueStakeSet,
    LeagueSummaryReceived,
    LeagueWeekClosed,
    RoundScored,
    Stake,
)

PREFIX = "demo-seed"
DEFAULT_MEMBERS = ("demo-hemdutt", "demo-ilan", "demo-dominick")
SCHED = Schedule()  # local days, as on a box (America/New_York)
# habit id -> (value on a day it is met, value on a day it is missed); targets and directions: DIRECTION
VALUES = {"sleep": (7.2, 4.8), "meals": (3, 1), "focus": (2.5, 0.8), "energy_drinks": (1, 3), "leave_ready": (0, 1),
          "sort_right": (100, 50)}
MISSES_ONE_IN = (11, 5, 3, 7, 4)  # by member position: the first has the best week, the third the worst
DIVERSION = (0.88, 0.72, 0.55, 0.80, 0.64)  # Waste Watch: recycled / recyclable, by member position
# dares, as config/league.yaml's presets: (slug, title, habit, at least this much); VALUES meets each on a met day
DARES = (("sleep-six", "Six hours of sleep", "sleep", 6.0), ("three-meals", "Three meals a day", "meals", 3.0),
         ("focus-two", "Two focused hours", "focus", 2.0), ("sort-it-right", "Sort it right", "sort_right", 80.0))
REPLY_S = 28  # a "free" comes back about this long after a proposal (measured between two boxes)
# table -> the column that says whose row it is (None: the circle's own)
TABLES = {"circles": None, "members": "id", "shares": "member_id", "habit_summaries": "member_id",
          "impact_counters": "member_id", "game_scores": "member_id", "agent_messages": "from_member",
          "events": "created_by", "rsvps": "member_id", "league_stakes": "set_by", "league_weeks": None,
          "league_challenges": "started_by", "presence": "member_id", "sealed_messages": "from_member"}


class SeedRefused(Exception):
    """Nothing was written. The message is the reason, safe to print."""


@dataclass
class Step:
    what: str
    rows: dict[str, list[str]]  # table -> the keys of the rows this step makes sure are there
    run: Callable[["Stores"], None]
    week: str | None = None  # a League declaration: the "circle|week" key it would take


@dataclass
class Stores:
    db: NeonDb
    shares: NeonShares
    plans: NeonCircleStore
    league: NeonLeague

    @classmethod
    def on(cls, db: NeonDb) -> "Stores":
        return cls(db, NeonShares(db), NeonCircleStore(db), NeonLeague(db, CircleBoard(db)))


@dataclass
class Plan:
    circle: str
    name: str
    members: dict[str, str]  # id -> display name, in ranking order
    end: date
    days: list[date]
    steps: list[Step] = field(default_factory=list)
    week_stamps: dict[str, datetime] = field(default_factory=dict)  # league_weeks key -> closed_at: how a week row is known as ours

    def rows(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {t: [] for t in TABLES}
        for step in self.steps:
            for table, keys in step.rows.items():
                out[table] += [k for k in keys if k not in out[table]]
        return {t: keys for t, keys in out.items() if keys}


@dataclass
class Found:
    table: str
    key: str
    owner: str | None
    circle: str | None
    extra: object = None  # circles: the name; league_weeks: closed_at


def display(member: str) -> str:
    return member.removeprefix("demo-").replace("-", " ").replace("_", " ").title()


def parse_members(text: str | None) -> dict[str, str]:
    """`a,b,c` or `a=Display Name,b`: explicit ids, used as they are. None: the three `demo-` members."""
    if not text:
        return {m: display(m) for m in DEFAULT_MEMBERS}
    out = {}
    for part in text.split(","):
        member, _, name = part.strip().partition("=")
        if member.strip():
            out[member.strip()] = name.strip() or display(member.strip())
    return out


def _missed(position: int, habit_index: int, days_back: int) -> bool:
    """`days_back`: 0 is the last day. The first member last missed sleep ten days back, the second four, the
    third two, and each habit a day apart from the one before it, so the streaks differ and the first leads."""
    every = MISSES_ONE_IN[position % len(MISSES_ONE_IN)]
    return (days_back + habit_index) % every == every - 1


def _at(day: date, hour: int, minute: int = 0, second: int = 0) -> datetime:
    return datetime.combine(day, time(hour, minute, second), SCHED.tz)


class _Collect:
    """The Outbox port, in memory: CircleSharing shapes each share exactly as on a box, and this keeps the row."""

    def __init__(self) -> None:
        self.rows: list[OutboxRow] = []

    def enqueue(self, circle_id, member_id, kind, payload, now, share_id=None):
        self.rows.append(OutboxRow(share_id, circle_id, member_id, kind, payload, now))

    def pending(self) -> list[OutboxRow]:
        return self.rows

    def mark_synced(self, share_id, now) -> None:
        pass


def _shares(plan: Plan) -> list[tuple[OutboxRow, DayClosed | CountUpdated | RoundScored]]:
    """Every share of every member, oldest first, with the event a box would have shared it from."""
    out = []
    for position, member in enumerate(plan.members):
        outbox = _Collect()
        sharing = CircleSharing(outbox, None, plan.circle, member, opted_in=True)
        events: list = []
        streak = dict.fromkeys(VALUES, 0)
        seen = recyclable = 0
        for n, day in enumerate(plan.days):
            closed_at = _at(day + timedelta(days=1), 0, 5)  # a habit day closes just after local midnight
            for h, (habit, (good, bad)) in enumerate(VALUES.items()):
                missed = _missed(position, h, len(plan.days) - 1 - n)
                value = float(bad if missed else good)
                op, target = DIRECTION[habit]
                met = value >= target if op == ">=" else value <= target
                streak[habit] = streak[habit] + 1 if met else 0
                events.append(DayClosed(event_id=f"habits-day-{habit}-{day.isoformat()}", ts=closed_at, habit_id=habit,
                                        day=day, met=met, value=value, streak=streak[habit]))
            recyclable += 5 + (n + position) % 3
            seen = recyclable + 3 * (n + 1)
            recycled = round(recyclable * DIVERSION[position % len(DIVERSION)])
            events.append(CountUpdated(event_id=f"{PREFIX}-count-{day.isoformat()}", ts=_at(day, 21), items_seen=seen,
                                       recyclable_seen=recyclable, recycled=recycled, contamination=n // 3 + position,
                                       diversion_rate=round(recycled / recyclable, 3)))
        for n, item in enumerate(("bottle", "can")):
            events.append(RoundScored(event_id=f"{PREFIX}-round-{n}", ts=_at(plan.days[-1], 19, 5 * n + position),
                                      round_id=f"{PREFIX}:{plan.circle}:{member}:r{n}", session_id=f"{PREFIX}:{plan.circle}:{member}",
                                      players=1, item=item, correct=not (position and n), seconds=5.0 + 2 * position + n,
                                      path_len=6 + position, score=150.0 - 20 * position - 35 * n))
        for e in sorted(events, key=lambda e: e.ts):
            {DayClosed: sharing.on_day_closed, CountUpdated: sharing.on_count_updated, RoundScored: sharing.on_round_scored}[type(e)](e)
            out.append((outbox.rows[-1], e))
    return sorted(out, key=lambda pair: (pair[0].created_at, pair[0].share_id))


def _share_rows(row: OutboxRow) -> dict[str, list[str]]:
    p, base = row.payload, {"circles": [row.circle_id], "members": [row.member_id], "shares": [row.share_id]}
    if row.kind == "habit_summary":
        return base | {"habit_summaries": [f"{row.member_id}|{p['habit_id']}|{p['day']}"]}
    if row.kind == "impact_count":
        return base | {"impact_counters": [row.member_id]}
    return base | {"game_scores": [p["round_id"]]}


def _league(plan: Plan, shares: list) -> tuple[LeagueStakeSet, list[LeagueWeekClosed], LeagueChallengeStarted, LeagueChallengeClosed | None]:
    """The stake, each week the summaries close and one dare with its result, declared by the real League
    projection from the same summaries the seed shares (config/league.yaml is the rule set)."""
    first = next(iter(plan.members))
    league = League(load_league(ROOT / "config" / "league.yaml"), plan.circle, first, dict(plan.members), SCHED)
    week = scoring.week_of(plan.end) - timedelta(days=7)  # the last complete week
    stake = LeagueStakeSet(event_id=f"{PREFIX}-stake-{plan.circle}-{week.isoformat()}", ts=_at(week, 9), circle_id=plan.circle,
                           week_start=week, stake=Stake(kind="pays_for_food"), set_by=first)
    # A five-day dare from config/league.yaml's presets: the first one somebody kept all five days, so it has a winner.
    dare_days = {(week + timedelta(days=n)).isoformat() for n in range(5)}
    kept = {(row.member_id, row.payload["habit_id"]) for row, _ in shares if row.kind == "habit_summary"}
    kept -= {(row.member_id, row.payload["habit_id"]) for row, _ in shares
             if row.kind == "habit_summary" and row.payload["day"] in dare_days and not row.payload["met"]}
    slug, title, habit, target = next((d for d in DARES if any(h == d[2] for _, h in kept)), DARES[0])
    dare = LeagueChallengeStarted(event_id=f"{PREFIX}-dare-{plan.circle}-{week.isoformat()}", ts=_at(week, 9, 5),
                                  circle_id=plan.circle, challenge_id=f"{PREFIX}:{plan.circle}:{slug}:{week.isoformat()}",
                                  title=title, habit_id=habit, direction="at_least", target=target, starts=week,
                                  ends=week + timedelta(days=4), members=list(plan.members), started_by=first)
    feed: list = [stake, dare]
    for row, e in shares:
        if row.kind in ("habit_summary", "impact_count"):
            feed.append(LeagueSummaryReceived(event_id=f"{PREFIX}-sum-{row.share_id}", ts=e.ts, circle_id=plan.circle,
                                              member_id=row.member_id, kind=row.kind, payload=row.payload))
    declared: list = []
    for e in sorted(feed, key=lambda e: e.ts):
        if isinstance(e, LeagueStakeSet):
            league.on_stake_set(e)
        elif isinstance(e, LeagueChallengeStarted):
            declared += league.on_challenge_started(e)
        else:
            declared += league.on_summary_received(e)
    declared += league.flush(max(SCHED.day_start(plan.end), league.closes_at(week)))  # on a Monday too
    weeks = [e for e in declared if isinstance(e, LeagueWeekClosed)]
    done = next((e for e in declared if isinstance(e, LeagueChallengeClosed)), None)
    return stake, weeks, dare, done


def build(circle: str, name: str | None = None, members: dict[str, str] | None = None, days: int = 7,
          end: date | None = None) -> Plan:
    """Everything the seed writes, in order. Nothing is sent."""
    members = members or parse_members(None)
    if len(members) < 2:
        raise SeedRefused("at least two members: a plan with nobody invited would ask every real box in the circle")
    if not circle or days < 1 or days > 60:
        raise SeedRefused("--circle must be set and --days between 1 and 60")
    end = end or datetime.now(UTC).astimezone(SCHED.tz).date()
    first_day = min(end - timedelta(days=days), scoring.week_of(end) - timedelta(days=7))  # the whole last week, at least
    plan = Plan(circle, name or f"{circle} (demo seed)", members, end,
                [first_day + timedelta(days=n) for n in range((end - first_day).days)])
    if plan.name == circle:
        raise SeedRefused("--name must differ from the circle id: the name is how the seed knows a circle is its own")

    def make_circle(s: Stores) -> None:  # the shape of NeonDb.ensure's own statement, with the name
        s.db.run(lambda cur: cur.execute("INSERT INTO circles (id, name) VALUES (%s, %s) ON CONFLICT (id) DO NOTHING",
                                         (circle, plan.name)))

    plan.steps.append(Step(f"the circle, named {plan.name!r}", {"circles": [circle]}, make_circle))

    shares = _shares(plan)
    for row, _ in shares:
        plan.steps.append(Step(f"{row.kind} {row.share_id}", _share_rows(row), lambda s, row=row: s.shares.push(row)))

    ids = list(members)
    evening = [("dinner", "Dinner", ids[1], _at(end, 18, 30), timedelta(hours=1)),
               ("games", "Board game night", ids[0], _at(end, 20, 30), timedelta(hours=2))]
    for n, (slug, title, proposer, start, length) in enumerate(evening):
        thread = f"{PREFIX}:{circle}:{slug}:{end.isoformat()}"
        invitees = sorted(m for m in ids if m != proposer)
        proposal = Proposal(title=title, start=start, end=start + length, location="the Student Union", invitees=invitees)
        asked = _at(end - timedelta(days=1), 17, 10 * n)
        messages = [AgentMessage(id=f"{thread}:proposal", circle_id=circle, thread_id=thread, from_member=proposer,
                                 kind="proposal", proposal=proposal, created_at=asked)]
        messages += [AgentMessage(id=f"{thread}:reply:{m}", circle_id=circle, thread_id=thread, from_member=m, kind="reply",
                                  reply=Reply(answer="free"), created_at=asked + timedelta(seconds=REPLY_S * (k + 1)))
                     for k, m in enumerate(invitees)]
        for msg in messages:
            plan.steps.append(Step(f"{msg.kind} {msg.id}", {"circles": [circle], "members": [msg.from_member],
                                                              "agent_messages": [msg.id]}, lambda s, msg=msg: s.plans.post(msg)))
        plan.steps.append(Step(
            f"agreed: {title}, {start:%a %b %d %I:%M %p}", {"members": sorted(ids), "events": [thread],
                                                           "rsvps": [f"{thread}|{m}" for m in sorted(ids)]},
            lambda s, thread=thread, proposal=proposal, proposer=proposer: s.plans.record_event(circle, thread, proposal, proposer)))

    stake, weeks, dare, done = _league(plan, shares)
    key = f"{circle}|{stake.week_start.isoformat()}"
    plan.steps.append(Step(f"League stake for the week of {stake.week_start}", {"league_stakes": [key]},
                           lambda s: s.league.push_stake(stake), week=key))
    for w in weeks:
        key = f"{circle}|{w.week_start.isoformat()}"
        plan.week_stamps[key] = w.ts
        plan.steps.append(Step(f"League week of {w.week_start}: {w.summary}", {"league_weeks": [key]},
                               lambda s, w=w: s.league.push_week(w), week=key))
    plan.steps.append(Step(f"League dare: {done.summary if done else dare.title}", {"league_challenges": [dare.challenge_id]},
                           lambda s: s.league.push_challenge(dare, done)))
    return plan


# --- what is in the database (reads only) ---


def look(db: NeonDb, circle: str, members: list[str]) -> list[Found]:
    """Every row of this circle, and every row that belongs to one of `members` wherever it is."""
    queries = [
        ("circles", "SELECT id, NULL, id, name FROM circles WHERE id = %s", (circle,)),
        ("members", "SELECT id, id, circle_id, NULL FROM members WHERE circle_id = %s OR id = ANY(%s)", None),
        ("shares", "SELECT share_id, member_id, circle_id, NULL FROM shares WHERE circle_id = %s OR member_id = ANY(%s)", None),
        ("habit_summaries", "SELECT member_id || '|' || habit_id || '|' || day::text, member_id, circle_id, NULL"
                            " FROM habit_summaries WHERE circle_id = %s OR member_id = ANY(%s)", None),
        ("impact_counters", "SELECT member_id, member_id, circle_id, NULL FROM impact_counters"
                            " WHERE circle_id = %s OR member_id = ANY(%s)", None),
        ("game_scores", "SELECT round_id, member_id, circle_id, NULL FROM game_scores WHERE circle_id = %s OR member_id = ANY(%s)", None),
        ("agent_messages", "SELECT id, from_member, circle_id, NULL FROM agent_messages"
                           " WHERE circle_id = %s OR from_member = ANY(%s)", None),
        ("events", "SELECT id, created_by, circle_id, NULL FROM events WHERE circle_id = %s OR created_by = ANY(%s)", None),
        ("rsvps", "SELECT r.event_id || '|' || r.member_id, r.member_id, e.circle_id, NULL FROM rsvps r"
                  " JOIN events e ON e.id = r.event_id WHERE e.circle_id = %s OR r.member_id = ANY(%s)", None),
        ("league_stakes", "SELECT circle_id || '|' || week_start::text, set_by, circle_id, NULL FROM league_stakes"
                          " WHERE circle_id = %s OR set_by = ANY(%s)", None),
        ("league_weeks", "SELECT circle_id || '|' || week_start::text, NULL, circle_id, closed_at FROM league_weeks"
                         " WHERE circle_id = %s", (circle,)),
        ("league_challenges", "SELECT challenge_id, started_by, circle_id, NULL FROM league_challenges"
                              " WHERE circle_id = %s OR started_by = ANY(%s)", None),
        ("presence", "SELECT member_id, member_id, circle_id, NULL FROM presence WHERE circle_id = %s OR member_id = ANY(%s)", None),
        ("sealed_messages", "SELECT id, from_member, circle_id, NULL FROM sealed_messages"
                            " WHERE circle_id = %s OR from_member = ANY(%s)", None),
    ]

    def work(cur):
        out = []
        for table, sql, params in queries:
            cur.execute(sql, params or (circle, list(members)))
            out += [Found(table, key, owner, where, extra) for key, owner, where, extra in cur.fetchall()]
        return out

    return db.run(work)


@dataclass
class Verdict:
    ours: list[Found]  # rows in the database that are this plan's own
    others: list[Found]  # rows in this circle that are not
    skip: set[str]  # League week keys someone else already declared: left alone
    circle_is_ours: bool  # the circle is not there yet, or carries the seed's name
    refusal: str | None = None
    notes: list[str] = field(default_factory=list)


def judge(plan: Plan, found: list[Found], into_existing: bool = False) -> Verdict:
    planned = {(t, k) for t, keys in plan.rows().items() for k in keys}
    members = set(plan.members)

    def is_ours(f: Found) -> bool:
        if (f.table, f.key) not in planned:
            return False
        if f.table == "circles":
            return f.extra == plan.name
        if f.table == "league_weeks":
            return f.extra == plan.week_stamps.get(f.key)
        return f.owner in members and f.circle == plan.circle

    ours = [f for f in found if is_ours(f)]
    rest = [f for f in found if not is_ours(f)]
    v = Verdict(ours, [f for f in rest if f.circle == plan.circle and f.table != "circles"], set(),
                circle_is_ours=not any(f.table == "circles" for f in rest))
    elsewhere = sorted({f.owner for f in rest if f.table == "members" and f.owner in members and f.circle != plan.circle})
    strangers = sorted({f.owner for f in rest if f.table != "members" and f.owner in members})
    if elsewhere:
        v.refusal = (f"member id {', '.join(elsewhere)} already belongs to another circle. Member ids are global in the "
                     "database: choose other ids with --members.")
    elif strangers:
        v.refusal = (f"{', '.join(strangers)} already has rows this run would not write: a real box uses that id, or an "
                     "earlier seed with another --end, --days or --members is still there (remove that one first, with "
                     "its own values). Nothing of a real member is ever overwritten.")
    elif (v.others or not v.circle_is_ours) and not into_existing:
        who = sorted({f.owner for f in v.others if f.owner and f.owner not in members})
        v.refusal = (f"circle {plan.circle} already exists and is not this seed's"
                     + (f" (it has rows of {len(who)} other member(s))" if who else "")
                     + ". Pass --into-existing to add the demo members beside what is there, or choose another --circle.")
    taken = {f.key for f in v.others if f.table in ("league_stakes", "league_weeks")}
    v.skip = {s.week for s in plan.steps if s.week in taken}
    if v.skip and not v.refusal:
        v.notes.append(f"left alone, already declared by someone else: {len(v.skip)} League row(s) ({', '.join(sorted(v.skip))})")
    return v


# --- apply and remove ---


def manifest_path(data_dir: Path, circle: str) -> Path:
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in circle)
    return Path(data_dir) / "neon-seed" / f"{safe}.json"


def apply(plan: Plan, db: NeonDb, data_dir: Path, into_existing: bool = False) -> dict:
    """Write the plan. Returns {"new": rows per table that were not there, "verdict": Verdict}. Raises SeedRefused."""
    before = look(db, plan.circle, list(plan.members))
    verdict = judge(plan, before, into_existing)
    if verdict.refusal:
        raise SeedRefused(verdict.refusal)
    path = manifest_path(data_dir, plan.circle)
    old = json.loads(path.read_text()) if path.exists() else {}
    there = {f.key for f in before if f.table == "members"}
    manifest = {
        "circle": plan.circle, "name": plan.name, "members": plan.members, "end": plan.end.isoformat(),
        "days": len(plan.days), "first_day": plan.days[0].isoformat(),
        "created_circle": bool(old.get("created_circle")) or not any(f.table == "circles" for f in before),
        "created_members": sorted(set(old.get("created_members", [])) | (set(plan.members) - there)),
        "rows": plan.rows(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=1))  # before the first write: a run that stops half-way can be removed
    stores = Stores.on(db)
    for step in plan.steps:
        if step.week in verdict.skip or (step.rows.keys() == {"circles"} and not verdict.circle_is_ours):
            continue
        step.run(stores)
    had = {(f.table, f.key) for f in verdict.ours}
    after = judge(plan, look(db, plan.circle, list(plan.members)), into_existing=True)
    new: dict[str, int] = {}
    for f in after.ours:
        if (f.table, f.key) not in had:
            new[f.table] = new.get(f.table, 0) + 1
    return {"new": new, "verdict": after}


def removal(plan: Plan, verdict: Verdict, created_members: list[str] | None = None) -> list[tuple[str, tuple]]:
    """The statements that delete this plan's own rows: exact keys and their owner, never a whole table or circle."""
    c, members = plan.circle, list(plan.members)
    ours: dict[str, list[str]] = {}
    for f in verdict.ours:
        ours.setdefault(f.table, []).append(f.key)
    out: list[tuple[str, tuple]] = []

    def by_id(table: str, key: str, owner: str) -> None:
        if ours.get(table):
            out.append((f"DELETE FROM {table} WHERE circle_id = %s AND {key} = ANY(%s) AND {owner} = ANY(%s)",
                        (c, ours[table], members)))

    if ours.get("rsvps"):
        out.append(("DELETE FROM rsvps WHERE event_id = ANY(%s) AND member_id = ANY(%s)",
                    (sorted({k.split("|")[0] for k in ours["rsvps"]}), members)))
    by_id("events", "id", "created_by")
    by_id("agent_messages", "id", "from_member")
    by_id("shares", "share_id", "member_id")
    by_id("game_scores", "round_id", "member_id")
    by_id("league_challenges", "challenge_id", "started_by")
    for key in ours.get("habit_summaries", []):
        member, habit, day = key.split("|")
        out.append(("DELETE FROM habit_summaries WHERE circle_id = %s AND member_id = %s AND habit_id = %s AND day = %s",
                    (c, member, habit, date.fromisoformat(day))))
    if ours.get("impact_counters"):
        out.append(("DELETE FROM impact_counters WHERE circle_id = %s AND member_id = ANY(%s)", (c, ours["impact_counters"])))
    for key in ours.get("league_stakes", []):
        out.append(("DELETE FROM league_stakes WHERE circle_id = %s AND week_start = %s AND set_by = ANY(%s)",
                    (c, date.fromisoformat(key.split("|")[1]), members)))
    for key in ours.get("league_weeks", []):
        out.append(("DELETE FROM league_weeks WHERE circle_id = %s AND week_start = %s AND closed_at = %s",
                    (c, date.fromisoformat(key.split("|")[1]), plan.week_stamps[key])))
    # a member the seed made, and only once nothing at all refers to it (a member's rows go with it otherwise)
    mine = [m for m in ours.get("members", []) if m.startswith("demo-") or m in (created_members or [])]
    if mine:
        unused = " AND ".join(f"NOT EXISTS (SELECT 1 FROM {t} x WHERE x.{col} = m.id)"
                              for t, col in TABLES.items() if col and t != "members")
        out.append((f"DELETE FROM members m WHERE m.circle_id = %s AND m.id = ANY(%s) AND {unused}", (c, mine)))
    if any(f.table == "circles" for f in verdict.ours):  # it carries the seed's name; and only when it is empty
        empty = " AND ".join(f"NOT EXISTS (SELECT 1 FROM {t} x WHERE x.circle_id = c.id)" for t in TABLES if t not in ("circles", "rsvps"))
        out.append((f"DELETE FROM circles c WHERE c.id = %s AND c.name = %s AND {empty}", (c, plan.name)))
    return out


def remove(plan: Plan, db: NeonDb, data_dir: Path, created_members: list[str] | None = None) -> dict:
    """Delete what apply wrote, in one transaction. Returns {"removed": per table, "left": this plan's rows still there
    (a member something else refers to, a circle that is not empty), "others": rows of other members, untouched}."""
    saved = manifest_path(data_dir, plan.circle)
    if created_members is None and saved.exists():  # this machine's own record of the member rows the seed made
        created_members = json.loads(saved.read_text()).get("created_members")
    before = judge(plan, look(db, plan.circle, list(plan.members)), into_existing=True)
    statements = removal(plan, before, created_members)

    def work(cur):
        for sql, params in statements:
            cur.execute(sql, params)

    if statements:
        db.run(work)
    after = judge(plan, look(db, plan.circle, list(plan.members)), into_existing=True)
    still = {(f.table, f.key) for f in after.ours}
    removed: dict[str, int] = {}
    for f in before.ours:
        if (f.table, f.key) not in still:
            removed[f.table] = removed.get(f.table, 0) + 1
    if not after.ours:
        manifest_path(data_dir, plan.circle).unlink(missing_ok=True)
    return {"removed": removed, "left": after.ours, "others": after.others}


# --- the command ---


def _table(rows: dict[str, list[str]], ids: bool = True) -> list[str]:
    lines = []
    for table in TABLES:
        if rows.get(table):
            lines.append(f"  {table:18s} {len(rows[table]):4d}" + (f"  {', '.join(rows[table])}" if ids else ""))
    return lines


def _counts(counts: dict[str, int]) -> str:
    return ", ".join(f"{t} {counts[t]}" for t in TABLES if counts.get(t)) or "nothing"


def main(args, out=print, connect: Callable[[], object] | None = None) -> int:
    """`connect` (tests): the connection to use instead of the one NEON_DATABASE_URL names."""
    from contexts.circle import schema

    data_dir = Path(args.data_dir)
    saved = manifest_path(data_dir, args.circle)
    known = json.loads(saved.read_text()) if saved.exists() else {}
    if args.remove and known and not (args.end or args.members or args.name):
        # this machine wrote it: the manifest has the values the rows were made from
        args.end, args.name, members = date.fromisoformat(known["end"]), known["name"], known["members"]
        days = (date.fromisoformat(known["end"]) - date.fromisoformat(known["first_day"])).days if args.days is None else args.days
    else:
        members, days = parse_members(args.members), 7 if args.days is None else args.days
    try:
        plan = build(args.circle, args.name, members, days, args.end)
    except SeedRefused as refused:
        out(f"refused: {refused}")
        return 2
    rows = plan.rows()
    out(f"Circle {plan.circle} ({plan.name}); members " + ", ".join(f"{m} ({n})" for m, n in plan.members.items())
        + f"; days {plan.days[0]} to {plan.days[-1]}; plans on the evening of {plan.end}")

    url = None if connect else schema.url_from_env()
    db, transport = None, NeonTransport()  # this run's own choice of tcp or https (ATLAS_NEON_TRANSPORT)
    if connect or url:
        db = NeonDb(connect=connect or (lambda: transport.connect(url)), names=dict(plan.members), retry_s=0)
    try:
        if not args.apply:
            if args.remove:
                if db is None:
                    out("dry run of the removal: NEON_DATABASE_URL is not set, so the database was not looked at."
                        " It would delete, where they are there and the seed's own:")
                    for line in _table(rows):
                        out(line)
                    return 0
                verdict = judge(plan, look(db, plan.circle, list(plan.members)), into_existing=True)
                there: dict[str, list[str]] = {}
                for f in verdict.ours:
                    there.setdefault(f.table, []).append(f.key)
                out(f"dry run of the removal; nothing was changed. {len(verdict.ours)} of this seed's rows are in the database:")
                for line in _table(there):
                    out(line)
                out(f"other members' rows in this circle, never touched: {len(verdict.others)}")
                out("to delete them: the same command with --apply")
                return 0
            out(f"dry run; nothing was sent. --apply would make sure these {sum(len(k) for k in rows.values())} rows are there:")
            for line in _table(rows):
                out(line)
            if db is None:
                out("NEON_DATABASE_URL is not set: the database was not looked at.")
                return 0
            verdict = judge(plan, look(db, plan.circle, list(plan.members)), args.into_existing)
            out(f"in the database now: {len(verdict.ours)} of them; other members' rows in this circle: {len(verdict.others)}")
            for note in verdict.notes:
                out(note)
            if verdict.refusal:
                out(f"--apply would be refused: {verdict.refusal}")
                return 1
            return 0
        if db is None:
            out("NEON_DATABASE_URL is not set (environment or .env): nothing was done")
            return 2
        if args.remove:
            done = remove(plan, db, data_dir, known.get("created_members"))
            out(f"removed: {_counts(done['removed'])}")
            if done["left"]:
                out("still there (something else refers to it, so it was left): "
                    + ", ".join(f"{f.table} {f.key}" for f in done["left"]))
            out(f"other members' rows in this circle, untouched: {len(done['others'])}")
            return 0
        done = apply(plan, db, data_dir, args.into_existing)
        ours: dict[str, int] = {}
        for f in done["verdict"].ours:
            ours[f.table] = ours.get(f.table, 0) + 1
        out(f"wrote: {_counts(done['new'])}")
        out(f"in the database now, this seed's: {_counts(ours)}")
        for note in done["verdict"].notes:
            out(note)
        board = CircleBoard(db)
        latest: dict[str, dict] = {}
        for r in board.habit_leaderboard(plan.circle):
            if r["habit_id"] == "sleep":
                latest[r["member_id"]] = r
        out("habit_leaderboard (sleep streak): " + ", ".join(f"{r['display_name']} {r['streak']}" for r in latest.values()))
        out("impact_leaderboard (diverted): " + ", ".join(
            f"{r['display_name']} {r['diversion_rate']:.0%}" for r in board.impact_leaderboard(plan.circle)))
        out(f"manifest: {saved}")
        out(f"to remove from any machine: python -m contexts.demo neon --circle {plan.circle} --end {plan.end} "
            f"--days {len(plan.days)}" + (f" --members {args.members}" if args.members else "")
            + (f" --name {args.name!r}" if args.name else "") + " --remove --apply")
        return 0
    except SeedRefused as refused:
        out(f"refused, nothing was written: {refused}")
        return 1
    except CircleOffline as offline:  # its message is ours: the kind of failure, never the address
        out(f"the database could not be reached ({offline}); run it again: every step is safe to repeat")
        return 1
    except Exception as e:  # a driver's message can name the host: say only what kind it was
        code = getattr(e, "sqlstate", None)
        out(f"the database refused a statement ({type(e).__name__}{', SQLSTATE ' + code if code else ''});"
            " what was written so far is safe to repeat or to --remove")
        return 1
    finally:
        if transport.chosen:
            out("database reached over " + {"tcp": "tcp (psycopg, port 5432)", "https": "https (port 443)"}[transport.chosen])
        if db is not None:
            db.close()


def add_arguments(p) -> None:
    p.add_argument("--circle", required=True, help="the circle id to seed (a box sees it with ATLAS_CIRCLE_ID=<this>)")
    p.add_argument("--name", help="the circle's name; it marks the circle as the seed's own (default: '<id> (demo seed)')")
    p.add_argument("--members", help="explicit member ids, used as they are: a,b or a=Display Name,b "
                                     f"(default: {','.join(DEFAULT_MEMBERS)})")
    p.add_argument("--days", type=int, help="days of summaries before --end (default 7; always the whole last week too)")
    p.add_argument("--end", type=lambda s: datetime.fromisoformat(s).date(), help="the story's last day, YYYY-MM-DD (default: today)")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="print the rows and send nothing (the default)")
    mode.add_argument("--apply", action="store_true", help="write (or, with --remove, delete)")
    p.add_argument("--remove", action="store_true", help="delete exactly what --apply wrote, and nothing else")
    p.add_argument("--into-existing", action="store_true", help="add the demo members to a circle that has other rows")
    p.add_argument("--data-dir", type=Path, default=Path(os.environ.get("ATLAS_DATA_DIR", "data")),
                   help="where the manifest goes (<data-dir>/neon-seed/<circle>.json)")
