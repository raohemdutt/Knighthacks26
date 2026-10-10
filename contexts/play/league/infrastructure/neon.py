"""The League's rows in Neon (tables league_stakes, league_weeks, league_challenges in
contexts/circle/infrastructure/schema.sql). Reads go through the Circle's board; writes are
idempotent so the timer can repeat them. Not reached in tests: the fake remote stands in."""

import json
from datetime import date

from contexts.circle.infrastructure.neon import CircleBoard, NeonDb
from contracts.play import LeagueChallengeClosed, LeagueChallengeStarted, LeagueStakeSet, LeagueWeekClosed


class NeonLeague:
    def __init__(self, db: NeonDb, board: CircleBoard) -> None:
        self.db = db
        self.board = board

    def habit_summaries(self, circle_id: str, since: date) -> list[dict]:
        return self.board.habit_summaries(circle_id, since)

    def impact_counters(self, circle_id: str) -> list[dict]:
        return self.board.impact_counters(circle_id)

    def push_stake(self, e: LeagueStakeSet) -> None:
        def work(cur):
            self.db.ensure(cur, e.circle_id, e.set_by)
            cur.execute(
                "INSERT INTO league_stakes (circle_id, week_start, kind, text, set_by, set_at) VALUES (%s, %s, %s, %s, %s, %s)"
                " ON CONFLICT (circle_id, week_start) DO UPDATE SET kind = EXCLUDED.kind, text = EXCLUDED.text,"
                " set_by = EXCLUDED.set_by, set_at = EXCLUDED.set_at WHERE league_stakes.set_at <= EXCLUDED.set_at",
                (e.circle_id, e.week_start, e.stake.kind, e.stake.text, e.set_by, e.ts),
            )

        self.db.run(work)

    def push_week(self, e: LeagueWeekClosed) -> None:
        def work(cur):
            self.db.ensure(cur, e.circle_id)
            cur.execute(
                "INSERT INTO league_weeks (circle_id, week_start, week_end, standings, worst, best, stake, summary, closed_at)"
                " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT (circle_id, week_start) DO NOTHING",
                (e.circle_id, e.week_start, e.week_end, json.dumps([s.model_dump(mode="json") for s in e.standings]),
                 e.worst, e.best, json.dumps(e.stake.model_dump(mode="json")), e.summary, e.ts),
            )

        self.db.run(work)

    def push_challenge(self, started: LeagueChallengeStarted, closed: LeagueChallengeClosed | None) -> None:
        def work(cur):
            self.db.ensure(cur, started.circle_id, started.started_by)
            cur.execute(
                "INSERT INTO league_challenges (challenge_id, circle_id, title, habit_id, direction, target, starts, ends,"
                " members, started_by) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT (challenge_id) DO NOTHING",
                (started.challenge_id, started.circle_id, started.title, started.habit_id, started.direction, started.target,
                 started.starts, started.ends, json.dumps(started.members), started.started_by),
            )
            if closed is not None:
                cur.execute(
                    "UPDATE league_challenges SET results = %s, winners = %s, summary = %s, closed_at = %s"
                    " WHERE challenge_id = %s AND closed_at IS NULL",
                    (json.dumps([r.model_dump(mode="json") for r in closed.results]), json.dumps(closed.winners),
                     closed.summary, closed.ts, closed.challenge_id),
                )

        self.db.run(work)
