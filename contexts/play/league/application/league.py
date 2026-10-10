"""The League projection and its member actions.

State rebuilds from the log (every `on_*` handler below is replay-safe and subscribed with
replay=True):
    habits.day.closed, impact.count.updated       this box's own daily summaries and counts
    agent.action.proposed (Day Run commits)       the stars of each plan this box committed
    play.league.summary.received                  friends' shared rows (from Neon on the timer, or the demo seed)
    play.league.stake.set / challenge.started     what was approved
    play.league.week.closed / challenge.closed    what was already declared, so a restart never declares twice

The first event after a week boundary (event time, plus a short grace for the last summaries to
arrive) returns a `play.league.week.closed` event with a deterministic id; the bus drops the repeat
on replay. The follow-ups (a GroupMe post, a Circle plan) are proposed by the live-only handlers
and go nowhere without an approval. Member actions (set the stake, start a dare) raise approval
cards too; the executor runs only after the phone approves and only then publishes the state event.
"""

import re
from collections.abc import Awaitable, Callable
from datetime import date, datetime, time, timedelta
from uuid import uuid4

from contexts.play.league.config import LeagueConfig
from contexts.play.league.domain import scoring
from contexts.play.league.domain.scoring import Count, DaySummary, MemberWeek, PlanStars
from contracts.agent import ActionProposed
from contracts.circle import Proposal
from contracts.events import Event
from contracts.habits import DayClosed
from contracts.impact import CountUpdated
from contracts.play import (ChallengeResult, LeagueChallengeClosed, LeagueChallengeStarted, LeagueStakeSet,
                            LeagueSummaryReceived, LeagueWeekClosed, Stake)

CHAT_WRITE = "groupme.post"
DAYRUN_STARS = re.compile(r"Day Run block\(s\) .*\((\d+)-star plan\)")
LOCAL_CIRCLE = "league"  # the circle id when this box plays alone
MAX_INVITEES = 20
Publish = Callable[[Event], Awaitable[None]]


class League:
    def __init__(self, cfg: LeagueConfig, circle_id: str | None, me: str, names: dict[str, str], sched,
                 clock: Callable[[], datetime] | None = None, hub=None, circle=None, chat_group: str = "friends") -> None:
        self.cfg = cfg
        self.circle_id = circle_id or LOCAL_CIRCLE
        self.local = circle_id is None  # no Circle: this box plays alone from its own summaries
        self.me = me
        self.names = dict(names)
        self.names.setdefault(me, "You" if self.local else me)
        self.sched = sched  # local days (contexts.habits.domain.config.Schedule)
        self.clock = clock
        self.hub = hub  # live follow-ups: GroupMe through the connector hub, after approval
        self.circle = circle  # live follow-ups: the meal or outing as a Circle plan, after approval
        self.chat_group = chat_group
        self.fallback: Callable | None = None  # the circle_share executor for everything that is not the league's
        self.publish: Publish | None = None  # set by subscribe(): what the executor publishes with after an approval
        # The Impact story (contexts/impact), when the box has one: `week_brief(now)` -> this week's milestone and the
        # practice-set size, added to the team waste block. None: those two keys are None.
        self.impact = None
        # --- the projection ---
        self.habits: dict[tuple[str, str, date], DaySummary] = {}
        self.counts: dict[str, list[Count]] = {}
        self.plans: dict[tuple[str, date], PlanStars] = {}
        self.stakes: dict[date, Stake] = {}  # week_start -> stake
        self.closed: dict[date, LeagueWeekClosed] = {}
        self.challenges: dict[str, LeagueChallengeStarted] = {}
        self.challenge_results: dict[str, LeagueChallengeClosed] = {}
        self._first_day: date | None = None
        self._silently_closed: set[date] = set()
        self.last_ts: datetime | None = None

    # --- wiring ---

    def subscribe(self, bus) -> None:
        self.publish = bus.publish
        bus.subscribe("habits.day.closed", self.on_day_closed, replay=True)
        bus.subscribe("impact.count.updated", self.on_count_updated, replay=True)
        bus.subscribe("agent.action.proposed", self.on_action_proposed, replay=True)
        bus.subscribe("play.league.summary.received", self.on_summary_received, replay=True)
        bus.subscribe("play.league.stake.set", self.on_stake_set, replay=True)
        bus.subscribe("play.league.week.closed", self.on_week_closed, replay=True)
        bus.subscribe("play.league.challenge.started", self.on_challenge_started, replay=True)
        bus.subscribe("play.league.challenge.closed", self.on_challenge_closed, replay=True)
        bus.subscribe("", self.on_any, replay=True)  # the boundary: a date compare per event
        # Live only: the follow-ups are proposals (approval cards), never sends.
        bus.subscribe("play.league.week.closed", self.on_week_closed_live)
        bus.subscribe("play.league.challenge.closed", self.on_challenge_closed_live)

    # --- state from the log ---

    def on_day_closed(self, e: DayClosed) -> list[Event]:
        self._habit(DaySummary(self.me, e.habit_id, e.day, e.met, e.value))
        return self._advance(e.ts)

    def on_count_updated(self, e: CountUpdated) -> list[Event]:
        self._count(Count(self.me, e.ts, e.items_seen, e.recyclable_seen, e.recycled, e.contamination))
        return self._advance(e.ts)

    def on_action_proposed(self, e: ActionProposed) -> list[Event]:
        """Day Run stars: the only trace a committed plan leaves in the log is its calendar card."""
        if e.kind == "connector_write" and e.target == "calendar.create_event" and (m := DAYRUN_STARS.search(e.reason)):
            day = self.sched.local_day(e.ts)
            self.plans[(self.me, day)] = PlanStars(self.me, day, int(m[1]))
            self._saw(day)
        return self._advance(e.ts)

    def on_summary_received(self, e: LeagueSummaryReceived) -> list[Event]:
        if e.circle_id != self.circle_id:
            return []
        if e.display_name and e.member_id not in self.names:
            self.names[e.member_id] = e.display_name
        p = e.payload
        if e.kind == "habit_summary":
            self._habit(DaySummary(e.member_id, str(p["habit_id"]), date.fromisoformat(str(p["day"])), bool(p["met"]),
                                   float(p["value"])))
        elif e.kind == "impact_count":
            self._count(Count(e.member_id, e.ts, int(p.get("items_seen", 0)), int(p.get("recyclable_seen", 0)),
                              int(p.get("recycled", 0)), int(p.get("contamination", 0))))
        return self._advance(e.ts)

    def on_stake_set(self, e: LeagueStakeSet) -> None:
        if e.circle_id == self.circle_id:
            self.stakes[e.week_start] = e.stake

    def on_week_closed(self, e: LeagueWeekClosed) -> None:
        if e.circle_id == self.circle_id:
            self.closed[e.week_start] = e

    def on_challenge_started(self, e: LeagueChallengeStarted) -> list[Event]:
        if e.circle_id == self.circle_id:
            self.challenges[e.challenge_id] = e
        return self._advance(e.ts)

    def on_challenge_closed(self, e: LeagueChallengeClosed) -> None:
        if e.circle_id == self.circle_id:
            self.challenge_results[e.challenge_id] = e

    def on_any(self, e: Event) -> list[Event]:
        if e.type.startswith("play.league."):
            return []  # handled above; and a declaration must not declare the next thing from inside itself
        return self._advance(e.ts)

    def flush(self, now: datetime) -> list[Event]:
        """The ticker: close what the clock passed when no event did."""
        return self._advance(now)

    def _habit(self, s: DaySummary) -> None:
        self.habits[(s.member_id, s.habit_id, s.day)] = s
        self._saw(s.day)

    def _count(self, c: Count) -> None:
        rows = self.counts.setdefault(c.member_id, [])
        rows.append(c)
        rows.sort(key=lambda r: r.ts)
        self._saw(self.sched.local_day(c.ts))

    def _saw(self, day: date) -> None:
        if self._first_day is None or day < self._first_day:
            self._first_day = day

    # --- the boundary (event time) ---

    def closes_at(self, week_start: date) -> datetime:
        """When the week is declared: local Monday 00:00 after it, plus the grace for the last summaries."""
        return self.sched.day_start(week_start + timedelta(days=7)) + timedelta(hours=self.cfg.close_grace_hours)

    def _advance(self, ts: datetime) -> list[Event]:
        if self.last_ts is None or ts > self.last_ts:
            self.last_ts = ts
        out: list[Event] = []
        if self._first_day is not None:
            week = scoring.week_of(self._first_day)
            while self.closes_at(week) <= ts:
                if week not in self.closed and week not in self._silently_closed:
                    closed = self._close_week(week, ts)
                    if closed is None:
                        self._silently_closed.add(week)
                    else:
                        self.closed[week] = closed
                        out.append(closed)
                week += timedelta(days=7)
        for c in list(self.challenges.values()):
            if c.challenge_id not in self.challenge_results and self._challenge_closes_at(c) <= ts:
                closed = self._close_challenge(c, ts)
                self.challenge_results[c.challenge_id] = closed
                out.append(closed)
        return out

    def _close_week(self, week: date, ts: datetime) -> LeagueWeekClosed | None:
        weeks = self.week(week)
        if not any(w.score is not None for w in weeks):
            return None  # nothing judged: nothing to declare
        worst, best = scoring.worst_and_best(weeks)
        stake = self.stake_for(week)
        return LeagueWeekClosed(
            event_id=f"league-week-{self.circle_id}-{week.isoformat()}", ts=ts, circle_id=self.circle_id,
            week_start=week, week_end=scoring.week_end(week), standings=scoring.standings(weeks),
            worst=worst.member_id if worst else None, best=best.member_id if best else None, stake=stake,
            summary=scoring.summary_text(worst, best, stake),
        )

    def _challenge_closes_at(self, c: LeagueChallengeStarted) -> datetime:
        return self.sched.day_start(c.ends + timedelta(days=1)) + timedelta(hours=self.cfg.close_grace_hours)

    def _close_challenge(self, c: LeagueChallengeStarted, ts: datetime) -> LeagueChallengeClosed:
        results = [ChallengeResult(member_id=m, display_name=self.names.get(m, m), **self.challenge_progress(c, m, c.ends))
                   for m in c.members]
        winners = [r.member_id for r in results if r.done]
        days = (c.ends - c.starts).days + 1
        if winners:
            who = ", ".join(self.names.get(m, m) for m in winners)
            summary = f"{c.title}, {days} {'day' if days == 1 else 'days'}: {who} made it."
        else:
            summary = f"{c.title}, {days} {'day' if days == 1 else 'days'}: nobody made it all the way. Next time."
        return LeagueChallengeClosed(event_id=f"league-challenge-{c.challenge_id}-closed", ts=ts, circle_id=self.circle_id,
                                     challenge_id=c.challenge_id, title=c.title, results=results, winners=winners,
                                     summary=summary)

    def challenge_progress(self, c: LeagueChallengeStarted, member: str, until: date) -> dict:
        """met_days / days / done for one member, judged from the same daily summaries, up to `until`."""
        last = min(until, c.ends)
        met = judged = 0
        day = c.starts
        while day <= last:
            s = self.habits.get((member, c.habit_id, day))
            if s is not None:
                judged += 1
                met += int(s.value >= c.target if c.direction == "at_least" else s.value <= c.target)
            day += timedelta(days=1)
        days = (c.ends - c.starts).days + 1
        return {"met_days": met, "days": days, "done": met == days}

    # --- live follow-ups: approval cards, never sends ---

    def proposes_follow_ups(self, e: LeagueWeekClosed) -> bool:
        """One box speaks for the league, or every member's phone would get the same card: the best week's."""
        return self.local or e.best == self.me

    def on_week_closed_live(self, e: LeagueWeekClosed) -> list[ActionProposed]:
        if e.circle_id != self.circle_id or not self.proposes_follow_ups(e):
            return []
        out: list[ActionProposed] = []
        if self.hub is not None and self.hub.is_write(CHAT_WRITE):
            lines = [f"League, week of {e.week_start:%b} {e.week_start.day}: " + ", ".join(
                f"{s.rank}. {s.display_name} {s.score:.0f}%" if s.score is not None else f"{s.rank}. {s.display_name} (nothing judged)"
                for s in e.standings), e.summary]
            out.append(self.hub.propose_write(CHAT_WRITE, {"text": " ".join(lines), "group": self.chat_group},
                                              f"Post the League standings to the group chat: {e.summary}", e.ts))
        if self.circle is not None and e.stake.kind in ("pays_for_food", "picks_the_event") and e.best:
            names = {s.member_id: s.display_name for s in e.standings}
            best = names.get(e.best, e.best)
            worst = names.get(e.worst, e.worst) if e.worst else None
            if e.stake.kind == "pays_for_food" and worst:
                title = f"Dinner on {worst}, {best} picks the place"
            elif e.stake.kind == "picks_the_event":
                title = f"{best}'s pick: the next outing"
            else:
                return out
            start = datetime.combine(e.week_end + timedelta(days=5), time(19), self.sched.tz)  # the coming Friday evening
            invitees = [m for m in names if m != self.me]
            proposal = Proposal(title=title, start=start, end=start + timedelta(hours=2), invitees=invitees)
            out.append(self.circle.propose(proposal, f"The League's stake: ask the circle about \"{title}\""))
        return out

    def on_challenge_closed_live(self, e: LeagueChallengeClosed) -> list[ActionProposed]:
        c = self.challenges.get(e.challenge_id)
        if e.circle_id != self.circle_id or c is None or not (self.local or c.started_by == self.me):
            return []
        if self.hub is None or not self.hub.is_write(CHAT_WRITE):
            return []
        return [self.hub.propose_write(CHAT_WRITE, {"text": f"League dare over. {e.summary}", "group": self.chat_group},
                                       f"Post the dare's result to the group chat: {e.summary}", e.ts)]

    # --- member actions (approval cards) ---

    def propose_stake(self, stake: Stake, now: datetime) -> ActionProposed:
        if stake.kind == "custom" and not (stake.text or "").strip():
            raise ValueError("a custom stake needs its text")
        week = scoring.week_of(self.sched.local_day(now))
        return ActionProposed(
            ts=now, action_id=uuid4().hex, kind="circle_share", target=self.circle_id,
            reason=f"Set the League stake from the week of {week:%b} {week.day}: {scoring.stake_text(stake)}",
            params={"kind": "league_stake", "week_start": week.isoformat(), "stake": stake.model_dump(mode="json"),
                    "set_by": self.me},
        )

    def propose_challenge(self, kind: str, days: int, invitees: list[str], now: datetime,
                          target: float | None = None) -> ActionProposed:
        preset = self.cfg.presets.get(kind)
        if preset is None:
            raise ValueError(f"unknown dare {kind!r}; one of: {', '.join(sorted(self.cfg.presets))}")
        if not 1 <= days <= self.cfg.max_days:
            raise ValueError(f"days must be 1 to {self.cfg.max_days}")
        invitees = sorted({m.strip() for m in invitees if m and m.strip() != self.me})
        if len(invitees) > MAX_INVITEES:
            raise ValueError(f"at most {MAX_INVITEES} invitees")
        unknown = [m for m in invitees if m not in self.names]
        if unknown:
            raise ValueError(f"not in the circle: {', '.join(unknown)}")
        if target is not None and target < 0:
            raise ValueError("the target cannot be negative")
        starts = self.sched.local_day(now) + timedelta(days=1)  # tomorrow: a whole day is judged
        challenge = {
            "challenge_id": uuid4().hex, "title": preset.title, "habit_id": preset.habit, "direction": preset.direction,
            "target": preset.target if target is None else target, "starts": starts.isoformat(),
            "ends": (starts + timedelta(days=days - 1)).isoformat(), "members": sorted({self.me, *invitees}),
            "started_by": self.me,
        }
        who = ", ".join(self.names.get(m, m) for m in invitees) or "nobody else yet"
        return ActionProposed(
            ts=now, action_id=uuid4().hex, kind="circle_share", target=self.circle_id,
            reason=f"Start a League dare: {preset.title} for {days} {'day' if days == 1 else 'days'}, inviting {who}",
            params={"kind": "league_challenge", "challenge": challenge},
        )

    async def executor(self, req) -> str:
        """Workspace executor for circle_share. The league's kinds publish their state event here, after the
        approval; anything else goes to the Circle's own executor."""
        kind = req.params.get("kind")
        now = self.clock() if self.clock else (self.last_ts or datetime.now(self.sched.tz))
        if kind == "league_stake":
            stake = Stake.model_validate(req.params["stake"])
            e = LeagueStakeSet(ts=now, circle_id=self.circle_id, week_start=date.fromisoformat(req.params["week_start"]),
                               stake=stake, set_by=str(req.params.get("set_by") or self.me))
            await self._publish(e)
            return f"League stake set: {scoring.stake_text(stake)}"
        if kind == "league_challenge":
            e = LeagueChallengeStarted(ts=now, circle_id=self.circle_id, **req.params["challenge"])
            await self._publish(e)
            return f"League dare started: {e.title} until {e.ends:%a %b} {e.ends.day}"
        if self.fallback is not None:
            out = self.fallback(req)
            return await out if hasattr(out, "__await__") else out
        raise ValueError(f"unsupported share {kind}")

    async def _publish(self, e: Event) -> None:
        if self.publish is None:
            raise RuntimeError("the league is not wired to a bus")
        await self.publish(e)

    # --- reads ---

    def stake_for(self, week: date) -> Stake:
        """The stake set for that week, else the latest one set before it, else the config's."""
        earlier = [w for w in self.stakes if w <= week]
        return self.stakes[max(earlier)] if earlier else self.cfg.stake

    def week(self, week_start: date) -> list[MemberWeek]:
        counts = [c for rows in self.counts.values() for c in rows]
        return scoring.score_week(self.cfg, week_start, self.names, list(self.habits.values()), counts,
                                  list(self.plans.values()), self.sched.local_day)

    def team(self, week_start: date) -> scoring.TeamWaste:
        counts = [c for rows in self.counts.values() for c in rows]
        return scoring.team_waste(self.cfg, self.names, counts, week_start, self.sched.local_day)

    def impact_brief(self, now: datetime) -> dict:
        """This box's own week on the Impact screen: the latest milestone said and how many items are in the
        Sort It practice set. Additive keys; None when there is no Impact story or it cannot answer."""
        brief = {"milestone": None, "practice_size": None}
        if self.impact is not None:
            try:
                got = self.impact.week_brief(now)
                brief.update(milestone=got.get("milestone"), practice_size=got.get("practice_size"))
            except Exception:
                pass  # the standings never fail because the Impact story could not answer
        return brief

    def current_week(self, now: datetime) -> date:
        return scoring.week_of(self.sched.local_day(now))

    def view(self, now: datetime) -> dict:
        """GET /api/league: the running week, as the phone shows it."""
        week = self.current_week(now)
        weeks = self.week(week)
        worst, best = scoring.worst_and_best(weeks)
        stake = self.stake_for(week)
        closes_at = self.closes_at(week)
        team = self.team(week)
        last = max(self.closed.values(), key=lambda e: e.week_start, default=None)
        set_for = max((w for w in self.stakes if w <= week), default=None)
        return {
            "configured": True,
            "local": self.local,
            "circle_id": None if self.local else self.circle_id,
            "member_id": self.me,
            "note": ("This box plays alone: set circle.member_id and circle.circle_id in config/agent.yaml to play "
                     "with friends over the Circle.") if self.local else None,
            "members": [{"member_id": m, "display_name": n} for m, n in self.names.items()],
            "week": {"start": week.isoformat(), "end": scoring.week_end(week).isoformat(),
                     "closes_at": closes_at.isoformat(), "closes_in_s": max(0, int((closes_at - now).total_seconds()))},
            "targets": [{"id": t, "label": self.cfg.label(t), "weight": r.weight, "unit": r.unit, "source": r.source}
                        for t, r in self.cfg.targets.items()],
            "standings": [s.model_dump(mode="json") for s in scoring.standings(weeks)],
            "worst": _who(worst),
            "best": _who(best),
            "summary": scoring.summary_text(worst, best, stake),
            "stake": {**stake.model_dump(mode="json"), "text_for_week": scoring.stake_text(stake),
                      "set_for_week": set_for.isoformat() if set_for else None},
            "team_waste": {"line": team.line, "rate": team.rate, "target": team.target, "recycled": team.recycled,
                           "recyclable": team.recyclable, "members": team.members,
                           "contamination_of_the_week": team.contamination_of_the_week, **self.impact_brief(now)},
            "challenges": self.challenges_view(now),
            "presets": [{"kind": k, "title": p.title, "habit_id": p.habit, "direction": p.direction, "target": p.target}
                        for k, p in self.cfg.presets.items()],
            "last_closed": None if last is None else {"week_start": last.week_start.isoformat(), "worst": last.worst,
                                                      "best": last.best, "summary": last.summary},
        }

    def me_view(self, now: datetime) -> dict:
        week = self.current_week(now)
        mine = next((w for w in self.week(week) if w.member_id == self.me), None)
        return {
            "member_id": self.me,
            "display_name": self.names.get(self.me, self.me),
            "week": {"start": week.isoformat(), "end": scoring.week_end(week).isoformat()},
            "score": mine.score if mine else None,
            "met": mine.met if mine else 0,
            "judged": mine.judged if mine else 0,
            "reasons": mine.reasons() if mine else [],
            "breakdown": [{"target": t.target, "label": t.label, "unit": t.unit, "weight": t.weight, "met": t.met,
                           "judged": t.judged, "share": t.share, "detail": t.detail}
                          for t in (mine.targets.values() if mine else [])],
        }

    def challenges_view(self, now: datetime) -> list[dict]:
        today = self.sched.local_day(now)
        out = []
        for c in sorted(self.challenges.values(), key=lambda c: (c.starts, c.challenge_id)):
            result = self.challenge_results.get(c.challenge_id)
            status = "closed" if result else "upcoming" if today < c.starts else "running"
            members = []
            for m in c.members:
                p = self.challenge_progress(c, m, today)
                judged = sum(1 for d in range((min(today, c.ends) - c.starts).days + 1)
                             if (m, c.habit_id, c.starts + timedelta(days=d)) in self.habits) if today >= c.starts else 0
                members.append({"member_id": m, "display_name": self.names.get(m, m), "met_days": p["met_days"],
                                "judged_days": judged, "days": p["days"]})
            out.append({"challenge_id": c.challenge_id, "title": c.title, "habit_id": c.habit_id, "direction": c.direction,
                        "target": c.target, "starts": c.starts.isoformat(), "ends": c.ends.isoformat(),
                        "days": (c.ends - c.starts).days + 1, "started_by": c.started_by, "status": status,
                        "members": members, "winners": result.winners if result else [],
                        "summary": result.summary if result else None})
        return out


def _who(w: MemberWeek | None) -> dict | None:
    if w is None:
        return None
    return {"member_id": w.member_id, "display_name": w.display_name, "score": w.score, "reason": w.headline()}
