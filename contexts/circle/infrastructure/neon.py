"""Circle's shared state in Neon Postgres (N1). Tables: schema.sql in this folder.

    db = NeonDb()                      # reads NEON_DATABASE_URL; connects on first use
    store = NeonCircleStore(db)        # the CircleStore port (agent_messages, events, rsvps)
    shares = NeonShares(db)            # where the outbox worker pushes approved shares
    board = CircleBoard(db)            # leaderboards for the phone's Circle screen

The driver (psycopg 3, `pip install -e ".[neon]"`) is imported on first use, so
the rest of the app runs without it. Every statement is safe to repeat, which
lets NeonDb reconnect and retry once when the connection has dropped. If that
fails too it raises CircleOffline.
"""

import json
import os
import threading
from collections.abc import Callable
from datetime import UTC, datetime

from contexts.circle.application.protocol import AgentMessage
from contexts.circle.application.sharing import CircleOffline, OutboxRow
from contracts.circle import Proposal

ENV_URL = "NEON_DATABASE_URL"
CONNECT_TIMEOUT_S = 10

MESSAGE_COLUMNS = "id, circle_id, thread_id, from_member, kind, proposal, reply, created_at"


def _connect_from_env():
    try:
        import psycopg
    except ImportError as e:
        raise CircleOffline('the Postgres driver is not installed: pip install -e ".[neon]"') from e
    url = os.environ.get(ENV_URL)
    if not url:
        raise CircleOffline(f"{ENV_URL} is not set")
    return psycopg.connect(url, connect_timeout=CONNECT_TIMEOUT_S)


def _dropped_errors() -> tuple[type[BaseException], ...]:
    """What a lost connection looks like: socket errors, plus the driver's own when it is installed."""
    try:
        import psycopg
    except ImportError:
        return (OSError,)
    return (OSError, psycopg.OperationalError, psycopg.InterfaceError)


class NeonDb:
    """One lazy connection, one transaction per `run`, shared by the store, the shares and the board."""

    def __init__(self, connect: Callable[[], object] | None = None, names: dict[str, str] | None = None) -> None:
        self._connect = connect or _connect_from_env
        self.names = names or {}  # member id -> display name, used when a member row is first created
        self._conn = None
        self._lock = threading.Lock()
        self.last_ok: datetime | None = None  # when a statement last went through (this process)
        self.offline: str | None = None  # why the last attempt failed; None once one succeeds

    def run(self, work: Callable):
        """Run `work(cursor)` in a transaction and return its result. Reconnects and retries once."""
        dropped = _dropped_errors()
        with self._lock:
            while True:
                reused = self._conn is not None
                try:
                    if self._conn is None:
                        self._conn = self._connect()
                    out = work(self._conn.cursor())
                    self._conn.commit()
                    self.last_ok, self.offline = datetime.now(UTC), None
                    return out
                except dropped as e:
                    self._close()
                    if not reused:  # a fresh connection failed too: we are offline
                        self.offline = f"Neon is unreachable ({type(e).__name__})"
                        raise CircleOffline(self.offline) from e
                except CircleOffline as e:  # no driver or no URL: nothing to roll back
                    self.offline = str(e)
                    raise
                except Exception:
                    if self._conn is not None:
                        self._conn.rollback()
                    raise

    def close(self) -> None:
        with self._lock:
            self._close()

    def _close(self) -> None:
        conn, self._conn = self._conn, None
        if conn is None:
            return
        try:
            conn.close()
        except Exception:
            pass  # it was already gone

    def ensure(self, cur, circle_id: str, *member_ids: str) -> None:
        """Foreign keys are enforced: make sure the circle and member rows exist. Never overwrites."""
        cur.execute("INSERT INTO circles (id, name) VALUES (%s, %s) ON CONFLICT (id) DO NOTHING", (circle_id, circle_id))
        for m in member_ids:
            cur.execute(
                "INSERT INTO members (id, circle_id, display_name, has_box) VALUES (%s, %s, %s, true)"
                " ON CONFLICT (id) DO NOTHING",
                (m, circle_id, self.names.get(m, m)),
            )


def _json(value):
    return json.loads(value) if isinstance(value, str) else value


def _message(row) -> AgentMessage:
    id_, circle_id, thread_id, from_member, kind, proposal, reply, created_at = row
    return AgentMessage(id=id_, circle_id=circle_id, thread_id=thread_id, from_member=from_member, kind=kind,
                        proposal=_json(proposal), reply=_json(reply), created_at=created_at)


class NeonCircleStore:
    """CircleStore on agent_messages / events / rsvps. Same behaviour as InMemoryCircleStore."""

    def __init__(self, db: NeonDb) -> None:
        self.db = db

    def post(self, msg: AgentMessage) -> None:
        def work(cur):
            self.db.ensure(cur, msg.circle_id, msg.from_member)
            cur.execute(
                f"INSERT INTO agent_messages ({MESSAGE_COLUMNS}) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
                " ON CONFLICT (id) DO NOTHING",
                (msg.id, msg.circle_id, msg.thread_id, msg.from_member, msg.kind,
                 msg.proposal and msg.proposal.model_dump_json(), msg.reply and msg.reply.model_dump_json(),
                 msg.created_at),
            )

        self.db.run(work)

    def since(self, circle_id: str, after: datetime | None) -> list[AgentMessage]:
        where, params = "circle_id = %s", [circle_id]
        if after is not None:
            where, params = where + " AND created_at > %s", [circle_id, after]
        return self._messages(where, params)

    def thread(self, circle_id: str, thread_id: str) -> list[AgentMessage]:
        return self._messages("circle_id = %s AND thread_id = %s", [circle_id, thread_id])

    def _messages(self, where: str, params: list) -> list[AgentMessage]:
        def work(cur):
            cur.execute(f"SELECT {MESSAGE_COLUMNS} FROM agent_messages WHERE {where} ORDER BY created_at, id", params)
            return cur.fetchall()

        return [_message(r) for r in self.db.run(work)]

    def record_event(self, circle_id: str, thread_id: str, proposal: Proposal, created_by: str) -> None:
        """The agreed event goes on the shared calendar once; the first record for a thread wins."""
        attendees = sorted({*proposal.invitees, created_by})

        def work(cur):
            self.db.ensure(cur, circle_id, *attendees)
            cur.execute(
                "INSERT INTO events (id, circle_id, thread_id, title, starts_at, ends_at, location, created_by)"
                " VALUES (%s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
                (thread_id, circle_id, thread_id, proposal.title, proposal.start, proposal.end, proposal.location,
                 created_by),
            )
            if cur.rowcount != 1:
                return  # already recorded
            for m in attendees:  # agreement means every invitee answered free
                cur.execute(
                    "INSERT INTO rsvps (event_id, member_id, answer) VALUES (%s, %s, 'free')"
                    " ON CONFLICT (event_id, member_id) DO NOTHING",
                    (thread_id, m),
                )

        self.db.run(work)


class NeonShares:
    """Where the outbox worker lands a share: `shares` first, then the table for its kind."""

    def __init__(self, db: NeonDb) -> None:
        self.db = db

    def push(self, row: OutboxRow) -> None:
        def work(cur):
            self.db.ensure(cur, row.circle_id, row.member_id)
            cur.execute(
                "INSERT INTO shares (share_id, circle_id, member_id, kind, payload, created_at)"
                " VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT (share_id) DO NOTHING",
                (row.share_id, row.circle_id, row.member_id, row.kind, json.dumps(row.payload), row.created_at),
            )
            fan_out = FAN_OUT.get(row.kind)
            if fan_out:
                fan_out(cur, row)

        self.db.run(work)


def _habit_summary(cur, row: OutboxRow) -> None:
    p = row.payload  # contracts.habits.DayClosed
    cur.execute(
        "INSERT INTO habit_summaries (member_id, circle_id, habit_id, day, met, value, streak)"
        " VALUES (%s, %s, %s, %s, %s, %s, %s)"
        " ON CONFLICT (member_id, habit_id, day) DO UPDATE"
        " SET met = EXCLUDED.met, value = EXCLUDED.value, streak = EXCLUDED.streak",
        (row.member_id, row.circle_id, p["habit_id"], p["day"], p["met"], p["value"], p["streak"]),
    )


def _impact_count(cur, row: OutboxRow) -> None:
    p = row.payload  # contracts.impact.CountUpdated; an older count never replaces a newer one
    cur.execute(
        "INSERT INTO impact_counters (member_id, circle_id, items_seen, recyclable_seen, recycled, contamination,"
        " diversion_rate, updated_at) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
        " ON CONFLICT (member_id) DO UPDATE"
        " SET items_seen = EXCLUDED.items_seen, recyclable_seen = EXCLUDED.recyclable_seen,"
        " recycled = EXCLUDED.recycled, contamination = EXCLUDED.contamination,"
        " diversion_rate = EXCLUDED.diversion_rate, updated_at = EXCLUDED.updated_at"
        " WHERE impact_counters.updated_at <= EXCLUDED.updated_at",
        (row.member_id, row.circle_id, p["items_seen"], p["recyclable_seen"], p["recycled"], p["contamination"],
         p["diversion_rate"], row.created_at),
    )


def _game_score(cur, row: OutboxRow) -> None:
    p = row.payload  # contracts.play.RoundScored
    cur.execute(
        "INSERT INTO game_scores (round_id, session_id, member_id, circle_id, item, players, correct, seconds,"
        " path_len, score, played_at) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
        " ON CONFLICT (round_id) DO NOTHING",
        (p["round_id"], p["session_id"], row.member_id, row.circle_id, p["item"], p["players"], p["correct"],
         p["seconds"], p["path_len"], p["score"], row.created_at),
    )


FAN_OUT = {"habit_summary": _habit_summary, "impact_count": _impact_count, "game_score": _game_score}


class CircleBoard:
    """Read side for the phone's Circle screen. Each row is a dict keyed by column name."""

    def __init__(self, db: NeonDb) -> None:
        self.db = db

    def habit_leaderboard(self, circle_id: str) -> list[dict]:
        """Each member's current streak per habit, longest first."""
        return self._rows(
            "SELECT member_id, display_name, habit_id, streak, day FROM habit_leaderboard"
            " WHERE circle_id = %s ORDER BY streak DESC, display_name, habit_id",
            (circle_id,),
        )

    def impact_leaderboard(self, circle_id: str) -> list[dict]:
        """Waste Watch totals per member, best diversion rate first."""
        return self._rows(
            "SELECT member_id, display_name, items_seen, recycled, contamination, diversion_rate"
            " FROM impact_leaderboard WHERE circle_id = %s ORDER BY diversion_rate DESC, recycled DESC, display_name",
            (circle_id,),
        )

    def top_scores(self, circle_id: str, limit: int = 10) -> list[dict]:
        """Best Sort It rounds in the circle."""
        return self._rows(
            "SELECT g.round_id, g.member_id, m.display_name, g.item, g.players, g.correct, g.seconds, g.score,"
            " g.played_at FROM game_scores g JOIN members m ON m.id = g.member_id"
            " WHERE g.circle_id = %s ORDER BY g.score DESC, g.played_at LIMIT %s",
            (circle_id, limit),
        )

    def habit_summaries(self, circle_id: str, since) -> list[dict]:
        """Every member's daily habit summaries from `since` (a date) on, for the League's weekly scoring."""
        return self._rows(
            "SELECT member_id, habit_id, day, met, value, streak FROM habit_summaries"
            " WHERE circle_id = %s AND day >= %s ORDER BY day, member_id, habit_id",
            (circle_id, since),
        )

    def impact_counters(self, circle_id: str) -> list[dict]:
        """Each member's latest Waste Watch totals, with when they were updated (the League's team line)."""
        return self._rows(
            "SELECT member_id, items_seen, recyclable_seen, recycled, contamination, diversion_rate, updated_at"
            " FROM impact_counters WHERE circle_id = %s ORDER BY member_id",
            (circle_id,),
        )

    def upcoming_events(self, circle_id: str, now: datetime) -> list[dict]:
        """The shared calendar: agreed events that have not ended, soonest first, one row per RSVP.

        An event without RSVPs still comes back once, with member_id, display_name and answer set to None."""
        return self._rows(
            "SELECT e.id, e.thread_id, e.title, e.starts_at, e.ends_at, e.location, e.created_by,"
            " r.member_id, m.display_name, r.answer"
            " FROM events e LEFT JOIN rsvps r ON r.event_id = e.id LEFT JOIN members m ON m.id = r.member_id"
            " WHERE e.circle_id = %s AND e.ends_at > %s ORDER BY e.starts_at, e.id, m.display_name, r.member_id",
            (circle_id, now),
        )

    def _rows(self, sql: str, params: tuple) -> list[dict]:
        def work(cur):
            cur.execute(sql, params)
            return [dict(zip([d[0] for d in cur.description], r)) for r in cur.fetchall()]

        return self.db.run(work)


class NeonPresence:
    """The `presence` table (G5): one row per member, latest share wins, rows expire."""

    def __init__(self, db: NeonDb) -> None:
        self.db = db

    def push(self, row: OutboxRow) -> None:
        p = row.payload

        def work(cur):
            self.db.ensure(cur, row.circle_id, row.member_id)
            cur.execute(
                "INSERT INTO presence (member_id, circle_id, place, source, shared_at, expires_at)"
                " VALUES (%s, %s, %s, %s, %s, %s)"
                " ON CONFLICT (member_id) DO UPDATE SET circle_id = EXCLUDED.circle_id, place = EXCLUDED.place,"
                " source = EXCLUDED.source, shared_at = EXCLUDED.shared_at, expires_at = EXCLUDED.expires_at"
                " WHERE presence.shared_at <= EXCLUDED.shared_at",
                (row.member_id, row.circle_id, p["place"], p["source"], p["shared_at"], p["expires_at"]),
            )

        self.db.run(work)

    def active(self, circle_id: str, now) -> list[dict]:
        def work(cur):
            cur.execute(
                "SELECT member_id, place, source, shared_at, expires_at FROM presence"
                " WHERE circle_id = %s AND expires_at > %s ORDER BY member_id",
                (circle_id, now),
            )
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, r, strict=True)) for r in cur.fetchall()]

        return self.db.run(work)
