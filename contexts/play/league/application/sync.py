"""Friends' rows in, the league's declarations out (live only, on the app's timer; needs Neon).

    sync = LeagueSync(remote, league)
    events = sync.sync(now)     # friends' habit_summaries / impact_counters -> play.league.summary.received

The projection stays pure: Neon rows become events, and only the events feed the standings. The
league's own declarations (stake, closed week, dare) are pushed to the league_* tables after they
are in the log; every push is safe to repeat.
"""

import hashlib
import json
import logging
from datetime import date, datetime, timedelta
from typing import Protocol

from contexts.play.league.application.league import League
from contracts.play import LeagueChallengeClosed, LeagueChallengeStarted, LeagueStakeSet, LeagueSummaryReceived, LeagueWeekClosed

log = logging.getLogger("atlas.league")
LOOKBACK_DAYS = 21


class LeagueRemote(Protocol):
    def habit_summaries(self, circle_id: str, since: date) -> list[dict]: ...

    def impact_counters(self, circle_id: str) -> list[dict]: ...

    def push_stake(self, e: LeagueStakeSet) -> None: ...

    def push_week(self, e: LeagueWeekClosed) -> None: ...

    def push_challenge(self, started: LeagueChallengeStarted, closed: LeagueChallengeClosed | None) -> None: ...


class LeagueSync:
    def __init__(self, remote: LeagueRemote, league: League) -> None:
        self.remote = remote
        self.league = league
        self._pushed: set[str] = set()

    def sync(self, now: datetime) -> list[LeagueSummaryReceived]:
        """Pull friends' rows the league has not seen; push declarations it has not sent. CircleOffline propagates."""
        out = self._pull(now)
        self._push()
        return out

    def _pull(self, now: datetime) -> list[LeagueSummaryReceived]:
        lg = self.league
        since = lg.sched.local_day(now) - timedelta(days=LOOKBACK_DAYS)
        out: list[LeagueSummaryReceived] = []
        for r in self.remote.habit_summaries(lg.circle_id, since):
            member = str(r["member_id"])
            if member == lg.me:
                continue
            day = r["day"] if isinstance(r["day"], date) else date.fromisoformat(str(r["day"]))
            payload = {"habit_id": str(r["habit_id"]), "day": day.isoformat(), "met": bool(r["met"]),
                       "value": float(r["value"]), "streak": int(r.get("streak") or 0)}
            known = lg.habits.get((member, payload["habit_id"], day))
            if known is not None and known.met == payload["met"] and known.value == payload["value"]:
                continue
            out.append(LeagueSummaryReceived(event_id=f"league-sum-{_digest(member, payload)}", ts=now, circle_id=lg.circle_id,
                                             member_id=member, kind="habit_summary", payload=payload))
        for r in self.remote.impact_counters(lg.circle_id):
            member = str(r["member_id"])
            if member == lg.me:
                continue
            updated = r.get("updated_at")
            ts = updated if isinstance(updated, datetime) else now
            payload = {k: int(r.get(k) or 0) for k in ("items_seen", "recyclable_seen", "recycled", "contamination")}
            payload["diversion_rate"] = float(r.get("diversion_rate") or 0)
            rows = lg.counts.get(member) or []
            if rows and (rows[-1].items_seen, rows[-1].recyclable_seen, rows[-1].recycled, rows[-1].contamination) == tuple(
                    payload[k] for k in ("items_seen", "recyclable_seen", "recycled", "contamination")):
                continue
            out.append(LeagueSummaryReceived(event_id=f"league-cnt-{_digest(member, payload)}", ts=ts, circle_id=lg.circle_id,
                                             member_id=member, kind="impact_count", payload=payload))
        return out

    def _push(self) -> None:
        lg = self.league
        for week, stake in lg.stakes.items():
            key = f"stake:{week}:{stake.kind}:{stake.text}"
            if key not in self._pushed:
                self.remote.push_stake(LeagueStakeSet(circle_id=lg.circle_id, week_start=week, stake=stake, set_by=lg.me))
                self._pushed.add(key)
        for e in lg.closed.values():
            if e.event_id not in self._pushed:
                self.remote.push_week(e)
                self._pushed.add(e.event_id)
        for c in lg.challenges.values():
            closed = lg.challenge_results.get(c.challenge_id)
            key = f"{c.challenge_id}:{'closed' if closed else 'open'}"
            if key not in self._pushed:
                self.remote.push_challenge(c, closed)
                self._pushed.add(key)


def _digest(member: str, payload: dict) -> str:
    return hashlib.sha256(json.dumps([member, payload], sort_keys=True).encode()).hexdigest()[:20]
