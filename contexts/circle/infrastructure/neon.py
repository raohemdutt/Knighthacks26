"""Circle's shared state in Neon Postgres (N1). Tables: schema.sql in this folder.

    db = NeonDb()                      # reads NEON_DATABASE_URL; connects on first use
    store = NeonCircleStore(db)        # the CircleStore port (agent_messages, events, rsvps)
    shares = NeonShares(db)            # where the outbox worker pushes approved shares
    board = CircleBoard(db)            # leaderboards for the phone's Circle screen

The driver (psycopg 3, `pip install -e ".[neon]"`) is imported on first use, so
the rest of the app runs without it. Every statement is safe to repeat, which
lets NeonDb reconnect and retry once when the connection has dropped. If that
fails too it raises CircleOffline.

Two ways to the same database, chosen by ATLAS_NEON_TRANSPORT (NeonTransport below):
- `tcp`: psycopg on Postgres's own port 5432;
- `https`: Neon's SQL endpoint on port 443 (neon_https.py; standard library only, the driver is not needed),
  for networks that refuse 5432. Its transactions are weaker: read "TRANSACTIONS" at the top of neon_https.py;
- `auto` (the default): tcp, and https for the rest of the process once 5432 cannot be reached.

A slow or unreachable database must never hold the box up:
- connecting gives up after CONNECT_TIMEOUT_S, and a connection that went dead is noticed by TCP keepalives
  (KEEPALIVE) instead of hanging for minutes; over https every request gives up after CONNECT_TIMEOUT_S;
- once offline, calls fail at once for RETRY_S instead of each waiting out the connect timeout, so one tick of
  the app pays for one attempt, not one per job;
- one statement runs at a time; a caller that cannot get its turn within BUSY_TIMEOUT_S gets CircleOffline
  rather than queueing behind a stuck one.
"""

import json
import logging
import os
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

from contexts.circle.application.protocol import AgentMessage
from contexts.circle.application.sharing import CircleOffline, OutboxRow
from contexts.circle.infrastructure.neon_https import HttpsConnection
from contracts.circle import Proposal

log = logging.getLogger("atlas.circle")

ENV_URL = "NEON_DATABASE_URL"
ENV_TRANSPORT = "ATLAS_NEON_TRANSPORT"  # auto (default) | tcp | https
CONNECT_TIMEOUT_S = 10
RETRY_S = 15  # after a failed attempt, fail fast this long (shorter than the app's tick, so every tick tries once)
BUSY_TIMEOUT_S = 5  # how long a caller waits for its turn on the one connection
# libpq client settings (never sent to the server, so the pooler does not care): probe an idle link after 10 s,
# three probes 5 s apart, and on Linux (the Jetson) give up on unacknowledged data after 20 s.
KEEPALIVE = {"keepalives": 1, "keepalives_idle": 10, "keepalives_interval": 5, "keepalives_count": 3,
             "tcp_user_timeout": 20000}

MESSAGE_COLUMNS = "id, circle_id, thread_id, from_member, kind, proposal, reply, created_at"


def _tcp_connect(url: str):
    """psycopg on port 5432. ImportError when the driver is not installed."""
    import psycopg

    return psycopg.connect(url, connect_timeout=CONNECT_TIMEOUT_S, **KEEPALIVE)


def _https_connect(url: str):
    return HttpsConnection(url, timeout=CONNECT_TIMEOUT_S)


class NeonTransport:
    """Which way to the database, and one log line saying so. Never logs any part of the connection string.

    `auto` tries psycopg first. When that cannot connect (refused, timed out, no route, or the driver is not
    installed) it uses HTTPS instead and remembers that until the process restarts: `chosen` stays "https" and
    port 5432 is not tried again, so a blocked port costs one failed attempt, not one per tick."""

    MODES = ("auto", "tcp", "https")

    def __init__(self, tcp: Callable[[str], object] = _tcp_connect, https: Callable[[str], object] = _https_connect) -> None:
        self._tcp, self._https = tcp, https
        self.chosen: str | None = None  # the transport of the last connection handed out
        self._said: str | None = None

    def connect(self, url: str, mode: str | None = None):
        mode = (mode if mode is not None else os.environ.get(ENV_TRANSPORT, "")).strip().lower() or "auto"
        if mode not in self.MODES:
            raise CircleOffline(f"{ENV_TRANSPORT} must be auto, tcp or https")
        if mode == "https" or (mode == "auto" and self.chosen == "https"):
            return self._use("https", "set by " + ENV_TRANSPORT if mode == "https" else "", url)
        try:
            conn = self._tcp(url)
        except ImportError as e:
            if mode == "tcp":
                raise CircleOffline('the Postgres driver is not installed: pip install -e ".[neon]"') from e
            return self._use("https", "the Postgres driver is not installed", url)
        except _dropped_errors() as e:
            if mode == "tcp":
                raise
            return self._use("https", f"port 5432 could not be reached: {type(e).__name__}", url)
        self._say("tcp", "")
        return conn

    def _use(self, name: str, why: str, url: str):
        conn = self._https(url)
        self._say(name, why)
        return conn

    def _say(self, name: str, why: str) -> None:
        self.chosen = name
        if self._said == name:
            return
        self._said = name
        port = "psycopg, port 5432" if name == "tcp" else "port 443"
        if why and not why.startswith("set by"):
            log.warning("Circle database: using %s (%s) until this process restarts; %s", name, port, why)
        else:
            log.info("Circle database: using %s (%s)%s", name, port, f", {why}" if why else "")


TRANSPORT = NeonTransport()  # the app's one: what it chose holds for the whole process


def _connect_from_env():
    url = os.environ.get(ENV_URL)
    if not url:
        raise CircleOffline(f"{ENV_URL} is not set")
    return TRANSPORT.connect(url)


def _dropped_errors() -> tuple[type[BaseException], ...]:
    """What a lost connection looks like: socket errors, plus the driver's own when it is installed."""
    try:
        import psycopg
    except ImportError:
        return (OSError,)
    return (OSError, psycopg.OperationalError, psycopg.InterfaceError)


class NeonDb:
    """One lazy connection, one transaction per `run`, shared by the store, the shares and the board.

    Over https "one transaction per `run`" holds for work that is one statement, or several writes with nothing
    read in between (they are sent together at the end). Work that reads an answer and then writes again is two
    transactions there; the two places that do it, record_event and NeonCleanup.sweep, say what that costs."""

    def __init__(self, connect: Callable[[], object] | None = None, names: dict[str, str] | None = None,
                 retry_s: float = RETRY_S, busy_timeout_s: float = BUSY_TIMEOUT_S,
                 monotonic: Callable[[], float] = time.monotonic) -> None:
        self._connect = connect or _connect_from_env
        self.names = names or {}  # member id -> display name, used when a member row is first created
        self._conn = None
        self._lock = threading.Lock()
        self.last_ok: datetime | None = None  # when a statement last went through (this process)
        self.offline: str | None = None  # why the last attempt failed; None once one succeeds
        self.retry_s, self.busy_timeout_s, self._monotonic = retry_s, busy_timeout_s, monotonic
        self._retry_at = 0.0  # while offline: no new attempt before this (monotonic seconds)

    def run(self, work: Callable):
        """Run `work(cursor)` in a transaction and return its result. Reconnects and retries once."""
        if self.offline and self._monotonic() < self._retry_at:
            raise CircleOffline(self.offline)  # it just failed: do not make every caller wait out the timeout again
        if not self._lock.acquire(timeout=self.busy_timeout_s):
            raise CircleOffline("Neon is slow to answer (an earlier statement is still waiting)")
        try:
            return self._run(work)
        except CircleOffline:
            self._retry_at = self._monotonic() + self.retry_s
            raise
        finally:
            self._lock.release()

    def _run(self, work: Callable):
        dropped = _dropped_errors()
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
                # a fresh connection failed too, or an https request already waited out its timeout: we are offline
                if not reused or not getattr(e, "retry", True):
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
        """The agreed event goes on the shared calendar once; the first record for a thread wins.

        Over https this is two transactions, because `rowcount` is read between the event and its RSVPs: the
        circle, member and event rows are committed first, the RSVP rows after. If the second request is lost,
        the event stays without RSVP rows, and a repeat does not add them (the event is "already recorded").
        The calendar then shows that event with nobody listed (upcoming_events returns it once, answer None).
        Over tcp it is one transaction and this cannot happen."""
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
    """The `presence` table (G5): one row per member, latest share wins, rows expire.

    Expiry runs on the database's clock, the one clock every box shares: a share is stored as "ends this long
    after now()" and read back as "this long left", so a box whose own clock is off neither hides a friend's
    place early nor shows it late. `shared_at` stays the sharer's own time: it only orders that member's shares."""

    def __init__(self, db: NeonDb) -> None:
        self.db = db

    def push(self, row: OutboxRow) -> None:
        p = row.payload

        def work(cur):
            self.db.ensure(cur, row.circle_id, row.member_id)
            cur.execute(
                "INSERT INTO presence (member_id, circle_id, place, source, shared_at, expires_at)"
                " VALUES (%s, %s, %s, %s, %s, now() + (%s::timestamptz - %s::timestamptz))"
                " ON CONFLICT (member_id) DO UPDATE SET circle_id = EXCLUDED.circle_id, place = EXCLUDED.place,"
                " source = EXCLUDED.source, shared_at = EXCLUDED.shared_at, expires_at = EXCLUDED.expires_at"
                " WHERE presence.shared_at <= EXCLUDED.shared_at",
                (row.member_id, row.circle_id, p["place"], p["source"], p["shared_at"], p["expires_at"],
                 p.get("sent_at") or p["shared_at"]),  # what is left of it when it is sent (nothing, for a stop)
            )

        self.db.run(work)

    def active(self, circle_id: str, now) -> list[dict]:
        def work(cur):
            cur.execute(
                "SELECT member_id, place, source, shared_at, %s::timestamptz + (expires_at - now()) AS expires_at"
                " FROM presence WHERE circle_id = %s AND expires_at > now() ORDER BY member_id",
                (now, circle_id),
            )
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, r, strict=True)) for r in cur.fetchall()]

        return self.db.run(work)


class NeonCleanup:
    """Keeps the shared tables from growing without bound. A box only ever removes rows it wrote itself.

    Per sweep, at most `limit` rows per table, oldest first:
    - `agent_messages` this member sent more than `keep` ago (the conversation view reads back 14 days);
    - `shares` this member pushed more than `keep` ago (the receipt; the summary rows they fed are kept);
    - this member's `presence` row once it has been expired for longer than `keep`.
    Agreed events, habit summaries, counters, scores and the League's weeks are history the Circle reads: kept."""

    def __init__(self, db: NeonDb, limit: int = 500) -> None:
        self.db = db
        self.limit = limit

    def sweep(self, circle_id: str, member_id: str, keep: timedelta) -> dict[str, int]:
        """Rows removed per table. `keep` is measured on the database's clock.

        Over https each of the three deletes is its own transaction (`rowcount` is read after each). Nothing
        depends on them happening together: a sweep that stops half-way has removed some old rows, the next
        one removes the rest; only the count in the log line is lost."""
        if keep <= timedelta(0):
            return {}

        def work(cur):
            removed = {}
            for table, key, owner in (("agent_messages", "id", "from_member"), ("shares", "share_id", "member_id")):
                cur.execute(
                    f"DELETE FROM {table} WHERE {key} IN (SELECT {key} FROM {table} WHERE circle_id = %s AND {owner} = %s"
                    " AND created_at < now() - %s ORDER BY created_at LIMIT %s)",
                    (circle_id, member_id, keep, self.limit),
                )
                removed[table] = cur.rowcount
            cur.execute("DELETE FROM presence WHERE circle_id = %s AND member_id = %s AND expires_at < now() - %s",
                        (circle_id, member_id, keep))
            removed["presence"] = cur.rowcount
            return removed

        return self.db.run(work)
